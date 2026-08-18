"""Run durable enrichment and entity jobs with bounded concurrency."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.db.engine import DatabaseResources, create_database_resources
from src.enrich_news.config import EnrichmentSettings, load_enrichment_settings
from src.enrich_news.models import NewsRecord
from src.enrich_news.pipeline import enrich_record, enrich_records
from src.enrich_news.provider import ClaudeProvider
from src.enrich_news.repository import EnrichmentRepository
from src.entity_bank.provider import ClaudeEntityProvider
from src.entity_bank.resolution_repository import ResolutionRepository
from src.entity_bank.resolver import CandidateIndex
from src.entity_bank.worker import Batch, process_market_events, process_news
from src.jobs.repository import (
    ENRICH_NEWS,
    RESOLVE_KALSHI_MARKET,
    RESOLVE_MARKET,
    RESOLVE_NEWS,
    SUPPORTED_JOB_TYPES,
    JobRecord,
    JobRepository,
)

WRITE_CONFIRMATION = "RUN_JOB_WORKER"
DEFAULT_CONCURRENCY = 10
MAX_CONCURRENCY = 30


@dataclass(frozen=True)
class JobResult:
    job_id: str
    job_type: str
    outcome: str
    details: dict[str, Any]


JobOutcome = JobResult | BaseException


def pack_job_groups(
    jobs: list[JobRecord],
    *,
    batch_size: int,
) -> list[list[JobRecord]]:
    """Group consecutive same-version enrich_news jobs up to ``batch_size``."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    groups: list[list[JobRecord]] = []
    pending: list[JobRecord] = []
    pending_version: str | None = None
    for job in jobs:
        version = str(job.payload.get("enrichment_version") or "")
        can_batch = job.job_type == ENRICH_NEWS and batch_size > 1
        if not can_batch:
            if pending:
                groups.append(pending)
                pending = []
                pending_version = None
            groups.append([job])
            continue
        if pending and (version != pending_version or len(pending) >= batch_size):
            groups.append(pending)
            pending = []
        if not pending:
            pending_version = version
        pending.append(job)
    if pending:
        groups.append(pending)
    return groups


class WorkerRuntime:
    def __init__(
        self,
        *,
        resources: DatabaseResources,
        settings: EnrichmentSettings,
        allow_network: bool,
        video_concurrency: int,
        market_concurrency: int,
    ) -> None:
        self.resources = resources
        self.settings = settings
        self.allow_network = allow_network
        self.enrichment_repository = EnrichmentRepository(resources)
        self.resolution_repository = ResolutionRepository(resources)
        self._thread_local = threading.local()
        self._candidate_lock = threading.Lock()
        self._bank_version_id: str | None = None
        self._candidate_index: CandidateIndex | None = None
        self.video_slots = threading.BoundedSemaphore(video_concurrency)
        self.market_slots = threading.BoundedSemaphore(market_concurrency)

    def _providers(self) -> tuple[ClaudeProvider, ClaudeEntityProvider]:
        providers = getattr(self._thread_local, "providers", None)
        if providers is None:
            api_key = self.settings.api_key or ""
            providers = (
                ClaudeProvider(
                    api_key,
                    model_name=self.settings.model_name,
                    max_tokens=self.settings.max_output_tokens,
                ),
                ClaudeEntityProvider(
                    api_key,
                    model_name=self.settings.model_name,
                    max_tokens=self.settings.max_output_tokens,
                ),
            )
            self._thread_local.providers = providers
        return providers

    def resolution_context(self) -> tuple[str, CandidateIndex]:
        latest_version = self.resolution_repository.latest_bank_version_id()
        if latest_version is None:
            raise RuntimeError("apply nflverse entity bank before processing resolution jobs")
        with self._candidate_lock:
            if (
                self._candidate_index is None
                or self._bank_version_id != latest_version
            ):
                self._candidate_index = CandidateIndex(
                    self.resolution_repository.load_candidate_rows()
                )
                self._bank_version_id = latest_version
            return latest_version, self._candidate_index

    def invalidate_candidates(self) -> None:
        with self._candidate_lock:
            self._candidate_index = None

    def handle(self, job: JobRecord) -> JobResult:
        if job.job_type == ENRICH_NEWS:
            return self._handle_enrich_news(job)
        if job.job_type == RESOLVE_NEWS:
            return self._handle_resolve_news(job)
        if job.job_type == RESOLVE_MARKET:
            return self._handle_resolve_market(job)
        if job.job_type == RESOLVE_KALSHI_MARKET:
            return self._handle_resolve_kalshi_market(job)
        raise ValueError(f"unsupported job type: {job.job_type}")

    def handle_group(self, jobs: list[JobRecord]) -> list[tuple[JobRecord, JobOutcome]]:
        if len(jobs) == 1:
            try:
                return [(jobs[0], self.handle(jobs[0]))]
            except Exception as exc:
                return [(jobs[0], exc)]
        try:
            return self._handle_enrich_news_batch(jobs)
        except Exception as exc:
            return [(job, exc) for job in jobs]

    def _handle_enrich_news_batch(
        self,
        jobs: list[JobRecord],
    ) -> list[tuple[JobRecord, JobOutcome]]:
        outcomes: list[tuple[JobRecord, JobOutcome]] = []
        pending: list[tuple[JobRecord, NewsRecord, str]] = []
        for job in jobs:
            news_id = str(job.payload["news_id"])
            version = str(
                job.payload.get("enrichment_version") or self.settings.enrichment_version
            )
            if self.enrichment_repository.has_completed(
                news_id=news_id,
                enrichment_version=version,
            ):
                outcomes.append(
                    (job, JobResult(job.job_id, job.job_type, "already_completed", {}))
                )
                continue
            record = self.enrichment_repository.load_record(news_id)
            if record is None:
                outcomes.append((job, ValueError(f"news row does not exist: {news_id}")))
                continue
            pending.append((job, record, version))
        if not pending:
            return outcomes

        provider, _ = self._providers()
        has_video = any(
            (media.media_type or "").casefold() in {"video", "animated_gif"}
            for _job, record, _version in pending
            for media in record.media
        )
        semaphore = self.video_slots if has_video else _NullSemaphore()
        version = pending[0][2]
        with semaphore:
            results = enrich_records(
                [record for _job, record, _version in pending],
                provider,
                enrichment_version=version,
                allow_network=self.allow_network,
            )
        for (job, _record, job_version), result in zip(pending, results, strict=True):
            if result.enrichment_version != job_version:
                result = result.model_copy(update={"enrichment_version": job_version})
            if not result.status.startswith("completed"):
                outcomes.append(
                    (job, RuntimeError(result.error or "news enrichment failed"))
                )
                continue
            self.enrichment_repository.persist_result(result)
            outcomes.append(
                (
                    job,
                    JobResult(
                        job.job_id,
                        job.job_type,
                        "completed",
                        {
                            "news_id": result.news_id,
                            "input_tokens": result.usage.input_tokens,
                            "output_tokens": result.usage.output_tokens,
                            "batch_size": len(pending),
                        },
                    ),
                )
            )
        return outcomes

    def _handle_enrich_news(self, job: JobRecord) -> JobResult:
        news_id = str(job.payload["news_id"])
        version = str(
            job.payload.get("enrichment_version")
            or self.settings.enrichment_version
        )
        if self.enrichment_repository.has_completed(
            news_id=news_id,
            enrichment_version=version,
        ):
            return JobResult(job.job_id, job.job_type, "already_completed", {})
        record = self.enrichment_repository.load_record(news_id)
        if record is None:
            raise ValueError(f"news row does not exist: {news_id}")
        provider, _ = self._providers()
        has_video = any(
            (media.media_type or "").casefold() in {"video", "animated_gif"}
            for media in record.media
        )
        semaphore = self.video_slots if has_video else _NullSemaphore()
        with semaphore:
            result = enrich_record(
                record,
                provider,
                enrichment_version=version,
                allow_network=self.allow_network,
            )
        if not result.status.startswith("completed"):
            raise RuntimeError(result.error or "news enrichment failed")
        self.enrichment_repository.persist_result(result)
        return JobResult(
            job.job_id,
            job.job_type,
            "completed",
            {
                "news_id": news_id,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
            },
        )

    def _handle_resolve_news(self, job: JobRecord) -> JobResult:
        bank_version_id, index = self.resolution_context()
        records = self.resolution_repository.load_news(
            limit=1,
            extractor_version=str(job.payload["extractor_version"]),
            enrichment_version=str(job.payload["enrichment_version"]),
            news_id=str(job.payload["news_id"]),
            input_fingerprint=str(job.payload["input_fingerprint"]),
        )
        if not records:
            return JobResult(job.job_id, job.job_type, "already_completed", {})
        _, provider = self._providers()
        batch = Batch()
        process_news(
            records=records,
            provider=provider,
            index=index,
            bank_version_id=bank_version_id,
            batch=batch,
            observed_at=datetime.now(UTC),
        )
        if batch.failures:
            raise RuntimeError(json.dumps(batch.failures, default=str))
        self.resolution_repository.persist_batch(batch.as_repository_batch())
        if batch.provisional_entities:
            self.invalidate_candidates()
        return JobResult(
            job.job_id,
            job.job_type,
            "completed",
            {
                "news_id": job.payload["news_id"],
                "mentions": len(batch.mentions),
                "input_tokens": batch.input_tokens,
                "output_tokens": batch.output_tokens,
            },
        )

    def _handle_resolve_market(self, job: JobRecord) -> JobResult:
        bank_version_id, index = self.resolution_context()
        events = self.resolution_repository.load_market_events(
            event_limit=1,
            event_ids={str(job.payload["event_id"])},
        )
        if not events:
            return JobResult(job.job_id, job.job_type, "not_active", {})
        _, provider = self._providers()
        batch = Batch()
        with self.market_slots:
            process_market_events(
                events=events,
                provider=provider,
                index=index,
                bank_version_id=bank_version_id,
                batch=batch,
                observed_at=datetime.now(UTC),
            )
        if batch.failures:
            raise RuntimeError(json.dumps(batch.failures, default=str))
        self.resolution_repository.persist_batch(batch.as_repository_batch())
        if batch.provisional_entities:
            self.invalidate_candidates()
        return JobResult(
            job.job_id,
            job.job_type,
            "completed",
            {
                "event_id": job.payload["event_id"],
                "markets": len(batch.classifications),
                "mentions": len(batch.mentions),
                "input_tokens": batch.input_tokens,
                "output_tokens": batch.output_tokens,
            },
        )

    def _handle_resolve_kalshi_market(self, job: JobRecord) -> JobResult:
        bank_version_id, index = self.resolution_context()
        events = self.resolution_repository.load_kalshi_market_events(
            event_limit=1,
            event_tickers={str(job.payload["event_ticker"])},
        )
        if not events:
            return JobResult(job.job_id, job.job_type, "not_active", {})
        _, provider = self._providers()
        batch = Batch()
        with self.market_slots:
            process_market_events(
                events=events,
                provider=provider,
                index=index,
                bank_version_id=bank_version_id,
                batch=batch,
                observed_at=datetime.now(UTC),
                source_kind="kalshi_market",
            )
        if batch.failures:
            raise RuntimeError(json.dumps(batch.failures, default=str))
        self.resolution_repository.persist_batch(batch.as_repository_batch())
        if batch.provisional_entities:
            self.invalidate_candidates()
        return JobResult(
            job.job_id,
            job.job_type,
            "completed",
            {
                "event_ticker": job.payload["event_ticker"],
                "markets": len(batch.classifications),
                "mentions": len(batch.mentions),
                "input_tokens": batch.input_tokens,
                "output_tokens": batch.output_tokens,
            },
        )


class _NullSemaphore:
    def __enter__(self) -> _NullSemaphore:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("JOB_WORKER_CONCURRENCY", DEFAULT_CONCURRENCY)),
    )
    parser.add_argument(
        "--video-concurrency",
        type=int,
        default=int(os.environ.get("JOB_VIDEO_CONCURRENCY", "2")),
    )
    parser.add_argument(
        "--market-concurrency",
        type=int,
        default=int(os.environ.get("JOB_MARKET_CONCURRENCY", "5")),
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=float(os.environ.get("JOB_POLL_INTERVAL_SECONDS", "1")),
    )
    parser.add_argument(
        "--lease-seconds",
        type=int,
        default=int(os.environ.get("JOB_LEASE_SECONDS", "900")),
    )
    parser.add_argument(
        "--job-types",
        default=",".join(sorted(SUPPORTED_JOB_TYPES)),
    )
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--confirm-live-writes")
    return parser


def _parse_job_types(value: str) -> set[str]:
    selected = {item.strip() for item in value.split(",") if item.strip()}
    unknown = selected - SUPPORTED_JOB_TYPES
    if not selected or unknown:
        raise ValueError(f"invalid job types: {sorted(unknown)}")
    return selected


def _log(event: str, **details: Any) -> None:
    print(
        json.dumps(
            {
                "event": event,
                "timestamp": datetime.now(UTC).isoformat(),
                **details,
            },
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.confirm_live_writes != WRITE_CONFIRMATION:
        print(
            f"ERROR: --confirm-live-writes must equal {WRITE_CONFIRMATION}",
            file=sys.stderr,
        )
        return 2
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        print(
            f"ERROR: --concurrency must be between 1 and {MAX_CONCURRENCY}",
            file=sys.stderr,
        )
        return 2
    args.video_concurrency = min(args.video_concurrency, args.concurrency)
    args.market_concurrency = min(args.market_concurrency, args.concurrency)
    if not 1 <= args.video_concurrency <= args.concurrency:
        print("ERROR: invalid --video-concurrency", file=sys.stderr)
        return 2
    if not 1 <= args.market_concurrency <= args.concurrency:
        print("ERROR: invalid --market-concurrency", file=sys.stderr)
        return 2
    if not 0.1 <= args.poll_interval_seconds <= 60:
        print("ERROR: invalid --poll-interval-seconds", file=sys.stderr)
        return 2
    if args.lease_seconds < 120:
        print("ERROR: --lease-seconds must be at least 120", file=sys.stderr)
        return 2
    try:
        job_types = _parse_job_types(args.job_types)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    src_dir = Path(__file__).resolve().parents[1]
    settings = load_enrichment_settings(src_dir)
    if not settings.api_key:
        print("ERROR: ANTHROPIC_API_KEY is not configured", file=sys.stderr)
        return 2
    resources = create_database_resources(src_dir)
    jobs = JobRepository(resources)
    runtime = WorkerRuntime(
        resources=resources,
        settings=settings,
        allow_network=not args.no_network,
        video_concurrency=args.video_concurrency,
        market_concurrency=args.market_concurrency,
    )
    lease_owner = f"{socket.gethostname()}:{os.getpid()}"
    stopping = threading.Event()

    def stop(*_args: object) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    futures: dict[
        Future[list[tuple[JobRecord, JobOutcome]]],
        list[JobRecord],
    ] = {}
    exit_code = 0
    _log(
        "JOB_WORKER_STARTED",
        concurrency=args.concurrency,
        video_concurrency=args.video_concurrency,
        market_concurrency=args.market_concurrency,
        enrichment_batch_size=settings.batch_size,
        job_types=sorted(job_types),
        lease_owner=lease_owner,
    )
    try:
        with ThreadPoolExecutor(
            max_workers=args.concurrency,
            thread_name_prefix="sports-job",
        ) as executor:
            while not stopping.is_set() or futures:
                done = {future for future in futures if future.done()}
                for future in done:
                    group = futures.pop(future)
                    try:
                        outcomes = future.result()
                    except Exception as exc:
                        outcomes = [(job, exc) for job in group]
                    for job, outcome in outcomes:
                        if isinstance(outcome, BaseException):
                            status = jobs.fail(
                                job,
                                lease_owner=lease_owner,
                                error=f"{type(outcome).__name__}: {outcome}",
                            )
                            _log(
                                "JOB_FAILED",
                                job_id=job.job_id,
                                job_type=job.job_type,
                                attempts=job.attempts,
                                status=status,
                                error=f"{type(outcome).__name__}: {outcome}",
                            )
                            if status == "dead":
                                exit_code = 1
                            continue
                        jobs.complete(job, lease_owner=lease_owner)
                        _log(
                            "JOB_COMPLETED",
                            job_id=outcome.job_id,
                            job_type=outcome.job_type,
                            outcome=outcome.outcome,
                            **outcome.details,
                        )

                if stopping.is_set():
                    if futures:
                        wait(
                            set(futures),
                            timeout=args.poll_interval_seconds,
                            return_when=FIRST_COMPLETED,
                        )
                    continue

                capacity = args.concurrency - len(futures)
                if capacity:
                    claim_limit = capacity
                    if ENRICH_NEWS in job_types:
                        claim_limit = capacity * settings.batch_size
                    claimed = jobs.claim(
                        limit=claim_limit,
                        lease_owner=lease_owner,
                        lease_seconds=args.lease_seconds,
                        job_types=job_types,
                    )
                    for group in pack_job_groups(
                        claimed,
                        batch_size=settings.batch_size,
                    ):
                        futures[executor.submit(runtime.handle_group, group)] = group

                if args.once and not futures and jobs.unfinished_count(
                    job_types=job_types
                ) == 0:
                    break
                if futures:
                    wait(
                        set(futures),
                        timeout=args.poll_interval_seconds,
                        return_when=FIRST_COMPLETED,
                    )
                else:
                    stopping.wait(args.poll_interval_seconds)
    finally:
        resources.close()
    _log("JOB_WORKER_STOPPED", exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
