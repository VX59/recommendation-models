from sqlalchemy.future import select
from sqlalchemy import desc
from sqlalchemy.orm import sessionmaker
from database.db import get_session
from database.models import MusiqlHistory, ModelUpdates

import numpy as np
from dataclasses import dataclass
import networkx as nx
from itertools import product
from functools import reduce
from typing import Optional

import logging

logger = logging.getLogger(__name__)



async def pull_history() -> Optional[list[MusiqlHistory]]:
    session_maker: sessionmaker = get_session()

    stmt = select(ModelUpdates.dttm).order_by(desc(ModelUpdates.dttm)).limit(1)

    async with session_maker() as session:
        result = await session.execute(stmt)
        last = result.scalar_one_or_none()

    if last:
        stmt = select(MusiqlHistory).where(MusiqlHistory.listened_at >= last)
    else:
        stmt = select(MusiqlHistory)

    async with session_maker() as session:
        result = await session.execute(stmt)
        rows = result.scalars().all()
        if rows:
            return rows

    return None


@dataclass
class SessionEvent:
    uri: str
    duration_played: float


def build_session(rows: list[MusiqlHistory]) -> list[SessionEvent]:
    session = []
    for event in rows:
        session.append(
            SessionEvent(
                event.uri,
                event.duration_played,
            )
        )
    return session


@dataclass
class ContinuityBreak:
    start: SessionEvent
    end: SessionEvent
    skipped_songs: list[SessionEvent]


def pull_continuous_session(
    session: list[SessionEvent],
) -> tuple[list[SessionEvent], list[ContinuityBreak]]:
    continuous_session = []

    full_play_threshold = 0.90

    for session_event in session:
        if session_event.duration_played >= full_play_threshold:
            continuous_session.append(session_event)

    continuity_gap = False
    K = None
    Ks = []
    for i in range(len(session)):
        if i > 0 and session[i].duration_played < full_play_threshold:
            if not continuity_gap:
                continuity_gap = True

                start_event = None
                if i > 0 and session[i - 1].duration_played >= full_play_threshold:
                    start_event = session[i - 1]

                K = ContinuityBreak(
                    start=start_event, end=None, skipped_songs=[session[i]]
                )
            else:
                K.skipped_songs.append(session[i])

        elif continuity_gap:
            continuity_gap = False
            K.end = session[i]
            Ks.append(K)

    if continuity_gap and K:
        Ks.append(K)

    return continuous_session, Ks


def build_session_dt_graph(session: list) -> nx.Graph:
    G = nx.DiGraph()
    for event in session:
        G.add_node(event.uri)

    return G


def Cs(delta_ij, n):
    return np.exp(-((delta_ij - 1) / np.sqrt(n)))


def calculate_Cs(G: nx.DiGraph, continuous_session: list[SessionEvent]):
    cs_ids = [event.uri for event in continuous_session]
    n = len(cs_ids)

    cs_index_map = {idx: uri for idx, uri in enumerate(cs_ids)}

    for i in range(n):
        for j in range(i + 1, n):
            delta_ij = j - i

            u = cs_index_map[i]
            v = cs_index_map[j]

            cs = Cs(delta_ij, n)

            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=0.0)

            G[u][v]["weight"] += cs


def Cw_penalty(K: int, S: int):
    return - (K / max(1, S * 0.1))


def calculate_Cw(G: nx.DiGraph, Ks: list[ContinuityBreak], S: int):
    for continuity_break in Ks:
        if continuity_break.skipped_songs:
            weighted_average = np.mean(
                [e.duration_played for e in continuity_break.skipped_songs]
            )
        else:
            continue

        cw_ids = [event.uri for event in continuity_break.skipped_songs]
        K = len(cw_ids)

        if K == 0:
            continue

        penalty = Cw_penalty(K, S) * (1 - weighted_average)

        if continuity_break.start and continuity_break.end:
            u = continuity_break.start.uri
            v = continuity_break.end.uri

            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=0.0)

            G[u][v]["weight"] += penalty

            for idx, skipped in enumerate(cw_ids):
                factor = (idx + 1) / K
                
                if not G.has_edge(u, skipped):
                    G.add_edge(u, skipped, weight=0.0)
                
                G[u][skipped]["weight"] += factor * penalty
                   
                if not G.has_edge(skipped, v):
                    G.add_edge(skipped, v, weight=0.0)

                G[skipped][v]["weight"] += factor * penalty


def historical_engagement(uri: str, session: list[SessionEvent]):
    finished_threshold = 0.90

    finished = 0
    num_played = 0

    for event in session:
        if event.duration_played >= finished_threshold and event.uri == uri:
            finished += 1
        if event.uri == uri:
            num_played += 1

    return (finished + 1) / (num_played + 2)


def calculate_Ce(G: nx.DiGraph, session: list[SessionEvent]):
    engagement = {
        u: historical_engagement(u, session)
        for u in G.nodes
    }

    for u in G.nodes:
        for v in G.successors(u):
            Ce = engagement[u] * engagement[v]

            if G.has_edge(u, v):
                G[u][v]["weight"] *= (1 + 0.37 * Ce)


async def compute_session() -> Optional[nx.DiGraph]:
    rows = await pull_history()
    if rows is None:
        return None

    session = build_session(rows)
    continuous_session, Ks = pull_continuous_session(session)

    Gdt: nx.DiGraph = build_session_dt_graph(session)

    calculate_Cw(Gdt, Ks, len(session))
    calculate_Cs(Gdt, continuous_session)
    calculate_Ce(Gdt, session)

    for u in Gdt.nodes:
        weights = [Gdt[u][v]["weight"] for v in Gdt.successors(u)]

        if not weights:
            continue

        min_w = min(weights)
        for v in Gdt.successors(u):
            Gdt[u][v]["weight"] -= min_w

        total = sum(Gdt[u][v]["weight"] for v in Gdt.successors(u))

        if total > 0:
            for v in Gdt.successors(u):
                Gdt[u][v]["weight"] /= total

    return Gdt
