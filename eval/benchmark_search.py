#!/usr/bin/env python
"""Benchmark dense search against hybrid search + RRF on a labeled QA set.

This runs the *real* retrieval code from ``services`` against a live
PostgreSQL + pgvector database and a real embedding provider; nothing here is
simulated. The corpus is the ``chunk_text`` of each entry in
``eval/qa_dataset.json``, so a small corpus of five documents isolates the effect
of the retrieval strategy rather than corpus size.

Usage::

    # First run: create the eval tenant/user and index the corpus.
    python eval/benchmark_search.py --seed

    # Subsequent runs: benchmark the existing corpus.
    python eval/benchmark_search.py

    python eval/benchmark_search.py --top-k 5 --output results.json

Requirements:
    * ``DATABASE_URL`` (or the POSTGRES_* settings) pointing at a pgvector DB
      that has been migrated with ``alembic upgrade head``.
    * ``OPENAI_API_KEY`` for query/document embeddings.

Metrics (higher is better):
    * recall@k - share of questions with at least one relevant chunk in the top k
    * MRR      - mean of 1/rank of the first relevant chunk (0 when none found)
    * latency  - mean wall-clock milliseconds per query
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any, Final
from uuid import NAMESPACE_URL, UUID, uuid5

# Allow running the script directly from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from config import get_settings
from database import get_engine, session_context
from models import Document, DocumentStatus, Tenant, User, UserRole
from rbac import accessible_min_roles
from schemas.documents import DocumentIngestRequest
from services.hybrid_search import hybrid_retrieve
from services.ingestion import DuplicateDocumentError, ingest_text
from services.retrieval import get_query_embedding, retrieve_chunks
from services.security import hash_password

DATASET_PATH: Final[Path] = Path(__file__).with_name("qa_dataset.json")

#: Deterministic ids make ``--seed`` idempotent across runs and machines.
EVAL_TENANT_ID: Final[UUID] = uuid5(NAMESPACE_URL, "https://atlas-rag.local/eval-tenant")
EVAL_USER_ID: Final[UUID] = uuid5(NAMESPACE_URL, "https://atlas-rag.local/eval-user")
EVAL_TENANT_SLUG: Final[str] = "eval-benchmark"
EVAL_USER_EMAIL: Final[str] = "eval@benchmark.local"
EVAL_USER_PASSWORD: Final[str] = "eval-only-password"


@dataclass(frozen=True, slots=True)
class QuestionResult:
    """Per-question retrieval outcome for one strategy."""

    question_id: int
    question: str
    dense_rank: int | None
    hybrid_rank: int | None
    dense_latency_ms: float
    hybrid_latency_ms: float

    @property
    def dense_reciprocal_rank(self) -> float:
        return 0.0 if self.dense_rank is None else 1.0 / self.dense_rank

    @property
    def hybrid_reciprocal_rank(self) -> float:
        return 0.0 if self.hybrid_rank is None else 1.0 / self.hybrid_rank

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "dense": {
                "first_relevant_rank": self.dense_rank,
                "reciprocal_rank": round(self.dense_reciprocal_rank, 4),
                "latency_ms": round(self.dense_latency_ms, 2),
            },
            "hybrid_rrf": {
                "first_relevant_rank": self.hybrid_rank,
                "reciprocal_rank": round(self.hybrid_reciprocal_rank, 4),
                "latency_ms": round(self.hybrid_latency_ms, 2),
            },
        }


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """Load and validate the labeled QA set."""
    with path.open(encoding="utf-8") as handle:
        dataset = json.load(handle)
    if not isinstance(dataset, list) or not dataset:
        raise ValueError(f"{path} must contain a non-empty JSON array")

    required = {"id", "title", "question", "chunk_text", "relevant_chunks"}
    for index, item in enumerate(dataset):
        missing = required - set(item)
        if missing:
            raise ValueError(f"Dataset entry {index} is missing {sorted(missing)}")
        if not item["relevant_chunks"]:
            raise ValueError(f"Dataset entry {index} has no relevance keywords")
    return dataset


def first_relevant_rank(contents: list[str], keywords: list[str]) -> int | None:
    """Return the 1-based rank of the first chunk containing all *keywords*."""
    needles = [keyword.lower() for keyword in keywords]
    for rank, content in enumerate(contents, start=1):
        haystack = content.lower()
        if all(needle in haystack for needle in needles):
            return rank
    return None


async def ensure_eval_identity(session: AsyncSession) -> tuple[Tenant, User]:
    """Create (or fetch) the deterministic eval tenant and user."""
    tenant = await session.get(Tenant, EVAL_TENANT_ID)
    if tenant is None:
        tenant = Tenant(
            id=EVAL_TENANT_ID, name="Retrieval Evaluation", slug=EVAL_TENANT_SLUG
        )
        session.add(tenant)
        await session.flush()

    user = await session.get(User, EVAL_USER_ID)
    if user is None:
        user = User(
            id=EVAL_USER_ID,
            tenant_id=EVAL_TENANT_ID,
            email=EVAL_USER_EMAIL,
            hashed_password=hash_password(EVAL_USER_PASSWORD),
            full_name="Retrieval Evaluator",
            role=UserRole.ADMIN,
            is_active=True,
        )
        session.add(user)
        await session.flush()
    return tenant, user


async def seed_corpus(
    session_factory: async_sessionmaker[AsyncSession], dataset: list[dict[str, Any]]
) -> int:
    """Index every dataset ``chunk_text`` into the eval tenant (idempotent)."""
    indexed = 0
    async with session_factory() as session:
        with session_context(tenant_id=EVAL_TENANT_ID, actor_id=EVAL_USER_ID):
            tenant, user = await ensure_eval_identity(session)
            await session.commit()

            for item in dataset:
                payload = DocumentIngestRequest(
                    title=str(item["title"]),
                    content=str(item["chunk_text"]),
                    source_system="evaluation",
                    mime_type="text/plain",
                    min_role_required=UserRole.INTERN,
                    metadata={"eval_id": item["id"]},
                )
                try:
                    await ingest_text(
                        session, tenant_id=tenant.id, actor=user, payload=payload
                    )
                except DuplicateDocumentError:
                    # Re-running --seed must not fail; the corpus is unchanged.
                    await session.rollback()
                else:
                    await session.commit()
                    indexed += 1
    return indexed


async def count_indexed_documents(session: AsyncSession) -> int:
    """Number of indexed documents visible to the eval tenant."""
    rows = await session.scalars(
        select(Document.id).where(
            Document.tenant_id == EVAL_TENANT_ID,
            Document.status == DocumentStatus.INDEXED,
            Document.deleted_at.is_(None),
        )
    )
    return len(rows.all())


async def benchmark(
    session_factory: async_sessionmaker[AsyncSession],
    dataset: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    """Run both strategies for every question and aggregate the metrics."""
    allowed_roles = accessible_min_roles(UserRole.ADMIN)
    results: list[QuestionResult] = []

    async with session_factory() as session:
        with session_context(tenant_id=EVAL_TENANT_ID, actor_id=EVAL_USER_ID):
            document_count = await count_indexed_documents(session)
            if document_count == 0:
                raise RuntimeError(
                    "The evaluation tenant has no indexed documents; run with --seed first"
                )

            for item in dataset:
                question = str(item["question"])
                keywords = [str(keyword) for keyword in item["relevant_chunks"]]
                embedding = await get_query_embedding(question)

                started = perf_counter()
                dense = await retrieve_chunks(
                    session,
                    tenant_id=EVAL_TENANT_ID,
                    query_embedding=embedding,
                    top_k=top_k,
                    allowed_roles=allowed_roles,
                )
                dense_ms = (perf_counter() - started) * 1000.0

                started = perf_counter()
                hybrid = await hybrid_retrieve(
                    session,
                    tenant_id=EVAL_TENANT_ID,
                    query_text=question,
                    query_embedding=embedding,
                    top_k=top_k,
                    allowed_roles=allowed_roles,
                )
                hybrid_ms = (perf_counter() - started) * 1000.0

                results.append(
                    QuestionResult(
                        question_id=int(item["id"]),
                        question=question,
                        dense_rank=first_relevant_rank(
                            [chunk.content for chunk in dense], keywords
                        ),
                        hybrid_rank=first_relevant_rank(
                            [chunk.content for chunk in hybrid], keywords
                        ),
                        dense_latency_ms=dense_ms,
                        hybrid_latency_ms=hybrid_ms,
                    )
                )

    def summarise(
        rank_attribute: str, reciprocal_attribute: str, latency_attribute: str
    ) -> dict[str, Any]:
        ranks = [getattr(result, rank_attribute) for result in results]
        reciprocal = [getattr(result, reciprocal_attribute) for result in results]
        return {
            "recall_at_k": round(sum(rank is not None for rank in ranks) / len(ranks), 4),
            "mrr": round(fmean(reciprocal), 4),
            "mean_latency_ms": round(
                fmean(getattr(result, latency_attribute) for result in results), 2
            ),
        }

    return {
        "corpus": {
            "tenant_id": str(EVAL_TENANT_ID),
            "indexed_documents": document_count,
            "questions": len(dataset),
            "top_k": top_k,
        },
        "dense": summarise("dense_rank", "dense_reciprocal_rank", "dense_latency_ms"),
        "hybrid_rrf": summarise(
            "hybrid_rank", "hybrid_reciprocal_rank", "hybrid_latency_ms"
        ),
        "per_question": [result.as_dict() for result in results],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Index the dataset corpus into the eval tenant before benchmarking",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Chunks retrieved per strategy (default: RETRIEVAL_TOP_K)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help=f"Labeled QA JSON (default: {DATASET_PATH.name})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Also write the JSON report to this path",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    if settings.openai_api_key is None and settings.llm_provider == "openai":
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "OPENAI_API_KEY is not configured; the benchmark needs embeddings.",
                file=sys.stderr,
            )
            return 2

    dataset = load_dataset(args.dataset)
    top_k = args.top_k or settings.retrieval_top_k

    engine = get_engine()
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        if args.seed:
            indexed = await seed_corpus(session_factory, dataset)
            print(f"Indexed {indexed} new document(s) into the eval tenant.")

        report = await benchmark(session_factory, dataset, top_k)
    finally:
        await engine.dispose()

    report["dataset"] = str(args.dataset)
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")

    dense, hybrid = report["dense"], report["hybrid_rrf"]
    print(
        "\nSummary"
        f"\n  dense      recall@{top_k}={dense['recall_at_k']:.2f} "
        f"MRR={dense['mrr']:.2f} {dense['mean_latency_ms']:.1f} ms"
        f"\n  hybrid+RRF recall@{top_k}={hybrid['recall_at_k']:.2f} "
        f"MRR={hybrid['mrr']:.2f} {hybrid['mean_latency_ms']:.1f} ms"
    )
    if hybrid["mrr"] < dense["mrr"]:
        print(
            "\nNote: hybrid MRR is below dense on this run. With a five-document "
            "corpus the variance is high; inspect per_question before drawing "
            "conclusions.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
