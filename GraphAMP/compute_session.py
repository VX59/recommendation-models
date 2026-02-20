from sqlalchemy.future import select
from db import async_session
from models import MusiqlHistory
import numpy as np
import asyncio
from dataclasses import dataclass
import networkx as nx
from itertools import product
import matplotlib.pyplot as plt
from functools import reduce
from tqdm import tqdm
import os
import joblib


async def pull_history() -> list[MusiqlHistory]:
    stmt = select(MusiqlHistory)
    async with async_session() as session:
        result = await session.execute(stmt)
        rows = result.scalars().all()
        if rows: return rows


@dataclass
class SessionEvent:
    uri:str
    duration_played:float


def build_session(rows:list[MusiqlHistory]) -> list[SessionEvent]:
    session = []
    for event in tqdm(rows):
        session.append(SessionEvent(
            event.uri,
            event.duration_played,
        ))
    return session


@dataclass
class ContinuityBreak:
    start:SessionEvent
    end:SessionEvent
    skipped_songs:list[SessionEvent]


def pull_continuous_session(session:list[SessionEvent]) -> tuple[list[SessionEvent], list[ContinuityBreak]]:
    continuous_session = []

    full_play_threshold = 0.99

    for session_event in session:
        if session_event.duration_played >= full_play_threshold: continuous_session.append(session_event)

    continuity_gap = False
    K = None
    Ks = []
    for i in tqdm(range(len(session))):
        if i > 0 and session[i].duration_played < full_play_threshold:
            if not continuity_gap:
                continuity_gap = True

                start_event = None
                if i > 0 and session[i-1].duration_played >= full_play_threshold:
                    start_event = session[i-1]

                K = ContinuityBreak(start=start_event, end=None, skipped_songs=[session[i]])
            else:
                K.skipped_songs.append(session[i])

        elif continuity_gap:
            continuity_gap = False
            K.end = session[i]
            Ks.append(K)

    if continuity_gap and K:
        Ks.append(K)

    return continuous_session, Ks


def build_session_dt_graph(session:list) -> nx.Graph:
    G = nx.DiGraph()
    for event in session:
        G.add_node(event.uri)

    uris = [event.uri for event in session]
    for i, j in product(uris, repeat=2):
        G.add_edge(i, j, weight=0.0)
        G.add_edge(j, i, weight=0.0)

    return G


def Cs(delta_ij,n):
    return np.exp(-((delta_ij-1)/np.sqrt(n)))


def calculate_Cs(G:nx.DiGraph, continuous_session:list[SessionEvent]):
    cs_ids = [event.uri for event in continuous_session]
    n = len(cs_ids)

    cs_index_map = {idx: uri for idx, uri in enumerate(cs_ids)}

    for i in tqdm(range(n)):
        for j in range(i+1, n):
            delta_ij = j - i
            
            u = cs_index_map[i]
            v = cs_index_map[j]
            
            G[u][v]["weight"] += Cs(delta_ij, n)
            

def Cw_penalty(K:int, S:int):
    return np.exp(-(np.pow(K,2))/(S))-1


def calculate_Cw(G:nx.DiGraph, Ks:list[ContinuityBreak], S:int):
    for continuity_break in Ks:

        weighted_average = np.average([event.duration_played for event in continuity_break.skipped_songs])
        cw_ids = [event.uri for event in continuity_break.skipped_songs]        
        K = len(cw_ids)
        
        if K == 0:
            continue

        if continuity_break.end:
            v = continuity_break.end.uri

            for u in cw_ids:
                G[u][v]["weight"] += Cw_penalty(K,S)*(1-weighted_average)

        if continuity_break.start:
            u = continuity_break.start.uri

            for v in cw_ids:
                G[u][v]["weight"] += Cw_penalty(K,S)*(1-weighted_average)
        

        for i,j in product(range(K),repeat=2):
            u = cw_ids[i]
            v = cw_ids[j]

            G[u][v]["weight"] += Cw_penalty(K,S)


def historical_engagement(uri:str, session:list[SessionEvent]):
    finished_threshold = 0.99

    def count_num_played(acc, event:SessionEvent):
        if event.uri == uri:
            acc += 1
        return acc

    def count_not_finished(acc, event:SessionEvent):
        if event.duration_played < finished_threshold and event.uri == uri:
            acc += 1
        return acc
    
    not_finished:float = reduce(count_not_finished, session, 0.0)
    num_played:float = reduce(count_num_played, session, 0.0)

    if num_played == 0:
        return 0

    return not_finished/num_played


def calculate_Ce(G:nx.DiGraph, session:list[SessionEvent]):
    for u,v in tqdm(product(G.nodes,repeat=2)):
        Ceu = historical_engagement(u, session)
        Cev = historical_engagement(v, session)
        Ce = Ceu * Cev

        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=0.0)
        if not G.has_edge(v, u):
            G.add_edge(v, u, weight=0.0)

        G[u][v]["weight"] += Ce
        G[v][u]["weight"] += Ce


async def main():
    rows = await pull_history()
    session = build_session(rows)
    continuous_session, Ks = pull_continuous_session(session)

    Gdt:nx.DiGraph = build_session_dt_graph(session)

    calculate_Cs(Gdt, continuous_session)
    calculate_Cw(Gdt, Ks, len(continuous_session))
    calculate_Ce(Gdt, session)

    file_name = f"recommendation-models/GraphAMP/models/GraphAMP_session.graph"
    
    os.makedirs("recommendation-models/GraphAMP/models", exist_ok=True)

    joblib.dump(Gdt, file_name, compress=3)

if __name__ == "__main__":
    asyncio.run(main())