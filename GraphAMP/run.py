from .accumulate import accumulate
from .compute_session import compute_session
from .seed import seed_new
from database.db import get_session
from s3_service import S3, get_S3
from database.models import Users, Models
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
import asyncio
from typing import List


async def main():
    session_maker:sessionmaker = get_session()
    async with session_maker() as session:
        stmt = (
            select(Models)
            .select_from(Users)
            .join(Models, Users.uri == Models.user_id)
            .where(Models.algorithm == "gamp")
        )
        results = await session.execute(stmt)        
        models:List[Models] = results.scalars().all()

    s3_service:S3 = get_S3()

    for model in models:
        G = await seed_new(model, s3_service)
        if not (Gdt := await compute_session()):
            continue
        await accumulate(
            G,
            Gdt,
            model,
            s3_service
        )
    

if __name__ == "__main__":
    asyncio.run(main())
