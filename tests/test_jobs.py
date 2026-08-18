from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

from sqlalchemy.dialects import postgresql

from src.db.engine import DatabaseResources
from src.enrich_news.config import EnrichmentSettings
from src.jobs.repository import (
    ENRICH_NEWS,
    RESOLVE_KALSHI_MARKET,
    RESOLVE_NEWS,
    SUPPORTED_JOB_TYPES,
    JobRecord,
    JobRepository,
    enqueue_job,
)
from src.jobs.worker import (
    DEFAULT_CONCURRENCY,
    MAX_CONCURRENCY,
    WorkerRuntime,
    pack_job_groups,
)


class FakeConnection:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.statements: list[Any] = []
        self.rows = rows or []

    def execute(self, statement: Any) -> Any:
        self.statements.append(statement)
        return FakeResult(self.rows)


class FakeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def mappings(self) -> "FakeResult":
        return self

    def all(self) -> list[dict[str, Any]]:
        return self.rows


class FakeBegin:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def __enter__(self) -> FakeConnection:
        return self.connection

    def __exit__(self, *_args: object) -> None:
        return None


class FakeEngine:
    def __init__(self, connection: FakeConnection | None = None) -> None:
        self.connection = connection or FakeConnection()

    def begin(self) -> FakeBegin:
        return FakeBegin(self.connection)

    def dispose(self) -> None:
        return None


def sql(statement: Any) -> str:
    return str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": False},
        )
    )


def test_resolve_kalshi_market_is_a_supported_job_type() -> None:
    assert RESOLVE_KALSHI_MARKET in SUPPORTED_JOB_TYPES


def test_enqueue_job_is_deterministic_and_transactional() -> None:
    connection = FakeConnection()

    first = enqueue_job(
        connection,
        job_type=ENRICH_NEWS,
        idempotency_key="x:123:v3",
        payload={"news_id": "x:123", "enrichment_version": "v3"},
    )
    second = enqueue_job(
        connection,
        job_type=ENRICH_NEWS,
        idempotency_key="x:123:v3",
        payload={"news_id": "x:123", "enrichment_version": "v3"},
    )

    assert first == second
    assert "ON CONFLICT (job_type, idempotency_key) DO NOTHING" in sql(
        connection.statements[0]
    )
    assert "pg_notify" in sql(connection.statements[1])


def test_enqueue_can_suppress_per_row_notifications_for_backfill() -> None:
    connection = FakeConnection()

    enqueue_job(
        connection,
        job_type=ENRICH_NEWS,
        idempotency_key="x:124:v3",
        payload={"news_id": "x:124"},
        notify=False,
    )

    assert len(connection.statements) == 1


def test_claim_is_atomic_and_uses_skip_locked() -> None:
    connection = FakeConnection(
        [
            {
                "job_id": "job-1",
                "job_type": ENRICH_NEWS,
                "idempotency_key": "x:125:v3",
                "payload": {"news_id": "x:125"},
                "attempts": 1,
                "max_attempts": 5,
            }
        ]
    )
    repository = JobRepository(
        DatabaseResources(engine=FakeEngine(connection))  # type: ignore[arg-type]
    )

    claimed = repository.claim(
        limit=10,
        lease_owner="worker-1",
        lease_seconds=900,
    )

    statement_sql = sql(connection.statements[0])
    assert "UPDATE job_outbox" in statement_sql
    assert "FOR UPDATE SKIP LOCKED" in statement_sql
    assert "job_outbox.attempts + " in statement_sql
    assert claimed[0].attempts == 1


def test_worker_uses_one_provider_pair_per_thread(monkeypatch: Any) -> None:
    created: list[tuple[str, int]] = []

    class Provider:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            created.append((type(self).__name__, id(self)))

    monkeypatch.setattr("src.jobs.worker.ClaudeProvider", Provider)
    monkeypatch.setattr("src.jobs.worker.ClaudeEntityProvider", Provider)
    settings = EnrichmentSettings(
        api_key="test-key",
        model_name="test-model",
        enrichment_version="v3",
        max_output_tokens=512,
        batch_size=2,
    )
    runtime = WorkerRuntime(
        resources=DatabaseResources(engine=FakeEngine()),  # type: ignore[arg-type]
        settings=settings,
        allow_network=False,
        video_concurrency=2,
        market_concurrency=4,
    )
    barrier = Barrier(4)

    def providers() -> tuple[Any, Any]:
        barrier.wait()
        first = runtime._providers()
        assert runtime._providers() is first
        return first

    with ThreadPoolExecutor(max_workers=4) as executor:
        pairs = list(executor.map(lambda _: providers(), range(4)))

    assert len({id(pair[0]) for pair in pairs}) == 4
    assert len({id(pair[1]) for pair in pairs}) == 4
    assert len(created) == 8
    assert DEFAULT_CONCURRENCY == 10
    assert MAX_CONCURRENCY == 30


def _job(
    job_id: str,
    *,
    job_type: str = ENRICH_NEWS,
    version: str = "v3",
) -> JobRecord:
    payload = {"news_id": job_id}
    if version:
        payload["enrichment_version"] = version
    return JobRecord(
        job_id=job_id,
        job_type=job_type,
        idempotency_key=job_id,
        payload=payload,
        attempts=1,
        max_attempts=5,
    )


def test_pack_job_groups_batches_same_version_enrich_news() -> None:
    groups = pack_job_groups(
        [
            _job("1"),
            _job("2"),
            _job("3"),
            _job("resolve", job_type=RESOLVE_NEWS),
            _job("4"),
            _job("5", version="v4"),
            _job("6", version="v4"),
        ],
        batch_size=2,
    )

    assert [[job.job_id for job in group] for group in groups] == [
        ["1", "2"],
        ["3"],
        ["resolve"],
        ["4"],
        ["5", "6"],
    ]


def test_pack_job_groups_keeps_singles_when_batch_size_is_one() -> None:
    groups = pack_job_groups([_job("1"), _job("2")], batch_size=1)
    assert [[job.job_id for job in group] for group in groups] == [["1"], ["2"]]
