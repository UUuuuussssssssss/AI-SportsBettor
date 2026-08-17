import gzip
import json
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

from sqlalchemy.dialects import postgresql

from src.linking.export_dataset import build_dataset_rows
from src.linking.labels import BucketRule, bucket_label, rule_column_name
from src.linking.linker import (
    LINKER_VERSION,
    LinkerRepository,
    MarketMention,
    NewsMention,
    build_links,
    market_final_before,
    market_open_at_publish,
)
from src.linking.reactions import (
    LABEL_VERSION,
    GcsEnvelopeReader,
    ReactionsRepository,
    build_reaction_rows,
    coverage_stats,
    select_envelopes,
)

NOW = datetime(2026, 8, 4, 18, 0, tzinfo=UTC)


def news_mention(**overrides: Any) -> NewsMention:
    values: dict[str, Any] = {
        "news_id": "x:200",
        "entity_id": "entity-1",
        "mention_role": "subject",
        "person_role_hint": "player",
        "published_at": NOW,
    }
    values.update(overrides)
    return NewsMention(**values)


def market_mention(**overrides: Any) -> MarketMention:
    values: dict[str, Any] = {
        "market_id": "m-1",
        "entity_id": "entity-1",
        "mention_role": "outcome",
        "person_role_hint": None,
        "event_id": "e-1",
        "closed_time": None,
        "resolution_observed_at": None,
        "first_observed_at": NOW - timedelta(days=3),
        "market_topic": "player_props",
        "contract_type": "binary",
    }
    values.update(overrides)
    return MarketMention(**values)


def test_shared_entity_produces_link_with_quality_features() -> None:
    rows = build_links(
        [
            news_mention(),
            news_mention(entity_id="entity-2", mention_role="mentioned"),
        ],
        [
            market_mention(),
            market_mention(entity_id="entity-2", mention_role="subject"),
        ],
    )

    assert len(rows) == 1
    row = rows[0]
    assert (row["news_id"], row["market_id"]) == ("x:200", "m-1")
    assert row["shared_entity_ids"] == ["entity-1", "entity-2"]
    assert row["shared_entity_count"] == 2
    assert row["news_mention_roles"]["entity-1"]["mention_roles"] == ["subject"]
    assert row["news_mention_roles"]["entity-1"]["person_role_hints"] == ["player"]
    assert row["market_mention_roles"]["entity-2"]["mention_roles"] == ["subject"]
    assert row["market_topic"] == "player_props"
    assert row["contract_type"] == "binary"
    assert row["market_open_at_publish"] is True
    assert row["linker_version"] == LINKER_VERSION


def test_disjoint_entities_do_not_link() -> None:
    rows = build_links(
        [news_mention(entity_id="entity-1")],
        [market_mention(entity_id="entity-9")],
    )
    assert rows == []


def test_market_final_before_publish_is_excluded() -> None:
    closed_before = market_mention(closed_time=NOW - timedelta(hours=1))
    resolved_before = market_mention(
        market_id="m-2",
        resolution_observed_at=NOW - timedelta(minutes=5),
    )
    closed_after = market_mention(
        market_id="m-3",
        closed_time=NOW + timedelta(hours=4),
    )

    assert market_final_before(closed_before, NOW) is True
    assert market_final_before(resolved_before, NOW) is True
    assert market_final_before(closed_after, NOW) is False

    rows = build_links(
        [news_mention()],
        [closed_before, resolved_before, closed_after],
    )
    assert [row["market_id"] for row in rows] == ["m-3"]


def test_market_discovered_after_publish_is_excluded() -> None:
    later = market_mention(first_observed_at=NOW + timedelta(days=1))
    unknown = market_mention(market_id="m-2", first_observed_at=None)
    assert market_open_at_publish(later, NOW) is False
    assert market_open_at_publish(unknown, NOW) is False
    rows = build_links([news_mention()], [later, unknown])
    assert rows == []


def test_links_are_deterministic_and_sorted() -> None:
    mentions = [
        news_mention(news_id="x:300", published_at=NOW - timedelta(hours=1)),
        news_mention(),
    ]
    markets = [
        market_mention(market_id="m-2"),
        market_mention(),
    ]
    first = build_links(mentions, markets)
    second = build_links(list(reversed(mentions)), list(reversed(markets)))
    assert first == second
    assert [(row["news_id"], row["market_id"]) for row in first] == [
        ("x:200", "m-1"),
        ("x:200", "m-2"),
        ("x:300", "m-1"),
        ("x:300", "m-2"),
    ]


class FakeResult:
    def __init__(self) -> None:
        self.rowcount = 0

    def __iter__(self) -> Any:
        return iter([])

    def mappings(self) -> list[Any]:
        return []


class FakeConnection:
    def __init__(self) -> None:
        self.statements: list[Any] = []

    def execute(self, statement: Any) -> FakeResult:
        self.statements.append(statement)
        return FakeResult()


class FakeTransaction:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def __enter__(self) -> FakeConnection:
        return self.connection

    def __exit__(self, *_args: object) -> None:
        return None


class FakeEngine:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def begin(self) -> FakeTransaction:
        return FakeTransaction(self.connection)

    def connect(self) -> FakeTransaction:
        return FakeTransaction(self.connection)


def compiled_sql(connection: FakeConnection) -> str:
    return "\n".join(
        str(statement.compile(dialect=postgresql.dialect()))
        for statement in connection.statements
    )


def test_linker_repository_upserts_without_pruning_by_default() -> None:
    connection = FakeConnection()
    resources = SimpleNamespace(engine=FakeEngine(connection), close=lambda: None)
    repository = LinkerRepository(resources)  # type: ignore[arg-type]
    rows = build_links([news_mention()], [market_mention()])

    repository.persist_links(rows, run_started_at=NOW)
    repository.finalize_run(run_started_at=NOW)

    sql = compiled_sql(connection)
    assert "INSERT INTO news_market_links" in sql
    assert "ON CONFLICT (news_id, market_id) DO UPDATE" in sql
    assert "updated_at <" not in sql
    assert "INSERT INTO ingest_cursors" in sql
    assert "excluded.last_successful_poll_at >= ingest_cursors.last_successful_poll_at" in sql


def test_full_rebuild_prunes_rows_not_regenerated() -> None:
    connection = FakeConnection()
    resources = SimpleNamespace(engine=FakeEngine(connection), close=lambda: None)
    repository = LinkerRepository(resources)  # type: ignore[arg-type]
    rows = build_links([news_mention()], [market_mention()])

    repository.persist_links(rows, run_started_at=NOW, prune=True)

    sql = compiled_sql(connection)
    assert "DELETE FROM news_market_links" in sql
    assert "news_market_links.updated_at <" in sql


def test_incremental_news_query_skips_already_linked_tweets() -> None:
    connection = FakeConnection()
    resources = SimpleNamespace(engine=FakeEngine(connection), close=lambda: None)
    repository = LinkerRepository(resources)  # type: ignore[arg-type]

    assert repository.load_news_mentions(since=NOW, skip_linked=True) == []

    sql = compiled_sql(connection)
    assert "news_market_links" in sql
    assert "news_entity_resolution_runs" in sql
    assert "news_events.published_at >" in sql


def test_delete_unopened_links_targets_flagged_rows() -> None:
    connection = FakeConnection()
    resources = SimpleNamespace(engine=FakeEngine(connection), close=lambda: None)
    repository = LinkerRepository(resources)  # type: ignore[arg-type]

    assert repository.delete_unopened_links() == 0
    sql = compiled_sql(connection)
    assert "DELETE FROM news_market_links" in sql
    assert "market_open_at_publish" in sql


def test_links_carry_their_platform() -> None:
    rows = build_links(
        [news_mention()],
        [
            market_mention(),
            market_mention(
                market_id="KXNFLGAME-26AUG15DALSEA-SEA",
                event_id="KXNFLGAME-26AUG15DALSEA",
                platform="kalshi",
            ),
        ],
    )
    platforms = {row["market_id"]: row["platform"] for row in rows}
    assert platforms == {
        "m-1": "polymarket",
        "KXNFLGAME-26AUG15DALSEA-SEA": "kalshi",
    }


def test_market_mentions_query_both_platforms() -> None:
    connection = FakeConnection()
    resources = SimpleNamespace(engine=FakeEngine(connection), close=lambda: None)
    repository = LinkerRepository(resources)  # type: ignore[arg-type]

    assert repository.load_market_mentions() == []

    sql = compiled_sql(connection)
    assert "entity_mentions.polymarket_market_id IS NOT NULL" in sql
    assert "entity_mentions.kalshi_market_ticker IS NOT NULL" in sql
    assert "JOIN kalshi_markets" in sql
    assert "kalshi_market_classifications" in sql
    assert "kalshi_markets.settlement_ts" in sql
    assert "kalshi_markets.close_time" in sql


def envelope_item(offset_seconds: int, run_id: str) -> dict[str, Any]:
    return {
        "ingest_run_id": run_id,
        "storage_uri": f"gs://bucket/{run_id}.json.gz",
        "ingested_at": NOW + timedelta(seconds=offset_seconds),
    }


def test_select_envelopes_prefers_newest_baseline_and_earliest_horizon() -> None:
    index = [
        envelope_item(-700, "too-old"),
        envelope_item(-300, "older-baseline"),
        envelope_item(-15, "baseline"),
        envelope_item(70, "plus-1m"),
        envelope_item(90, "later-1m"),
        envelope_item(302, "plus-5m"),
    ]
    chosen = select_envelopes(index, NOW)

    assert chosen["baseline"]["ingest_run_id"] == "baseline"
    assert chosen["plus_1m"]["ingest_run_id"] == "plus-1m"
    assert chosen["plus_5m"]["ingest_run_id"] == "plus-5m"
    assert chosen["plus_30m"] is None
    assert chosen["plus_2h"] is None


def test_select_envelopes_rejects_snapshots_beyond_tolerance() -> None:
    # First snapshot after the +1m horizon arrives four minutes late; a stale
    # collector gap must not masquerade as the one-minute price, while the
    # same snapshot legitimately serves the +5m horizon.
    index = [envelope_item(-15, "baseline"), envelope_item(300, "late")]
    chosen = select_envelopes(index, NOW)
    assert chosen["baseline"]["ingest_run_id"] == "baseline"
    assert chosen["plus_1m"] is None
    assert chosen["plus_5m"]["ingest_run_id"] == "late"


def test_coverage_stats_report_gaps_including_window_edges() -> None:
    index = [
        envelope_item(-15, "before"),
        envelope_item(30, "a"),
        envelope_item(45, "b"),
        envelope_item(3600, "c"),
    ]
    count, max_gap = coverage_stats(index, NOW)
    assert count == 3
    assert max_gap == Decimal("3600")

    empty_count, empty_gap = coverage_stats([envelope_item(-15, "before")], NOW)
    assert empty_count == 0
    assert empty_gap is None


def book(
    midpoint: str | None,
    *,
    spread: str | None = "0.02",
    bid_depth: str | None = "500",
    ask_depth: str | None = "500",
) -> dict[str, str | None]:
    return {
        "midpoint": midpoint,
        "spread": spread,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
    }


class FakeReader:
    def __init__(self, books: dict[str, dict[str, dict[str, str | None]]]) -> None:
        self.books = books
        self.fetches: list[str] = []

    def fetch_books(
        self, ingest_run_id: str, _storage_uri: str
    ) -> dict[str, dict[str, str | None]]:
        self.fetches.append(ingest_run_id)
        return self.books[ingest_run_id]


def test_reaction_rows_compute_deltas_and_coverage() -> None:
    link = {"news_id": "x:200", "market_id": "m-1", "published_at": NOW}
    tokens = [
        {"token_id": "t-yes", "outcome_index": 0},
        {"token_id": "t-no", "outcome_index": 1},
    ]
    index = [
        envelope_item(-15, "baseline"),
        envelope_item(70, "plus-1m"),
    ]
    reader = FakeReader(
        {
            "baseline": {
                "t-yes": book("0.40", spread="0.01", bid_depth="350", ask_depth="410"),
                "t-no": book("0.60"),
            },
            "plus-1m": {"t-yes": book("0.55")},
        }
    )

    rows = build_reaction_rows(
        link=link,
        tokens=tokens,
        envelope_index=index,
        reader=reader,
        trade_stats={"t-yes": (3, Decimal("120.5"))},
        computed_at=NOW + timedelta(hours=3),
    )

    assert len(rows) == 2
    yes_row = rows[0]
    assert yes_row["label_version"] == LABEL_VERSION
    assert yes_row["baseline_midpoint"] == Decimal("0.40")
    assert yes_row["baseline_spread"] == Decimal("0.01")
    assert yes_row["baseline_bid_depth"] == Decimal("350")
    assert yes_row["baseline_ask_depth"] == Decimal("410")
    assert yes_row["midpoint_plus_1m"] == Decimal("0.55")
    assert yes_row["delta_plus_1m"] == Decimal("0.15")
    assert yes_row["midpoint_plus_2h"] is None
    assert yes_row["delta_plus_2h"] is None
    assert yes_row["trade_count"] == 3
    assert yes_row["trade_notional"] == Decimal("120.5")
    assert yes_row["snapshot_count"] == 1

    no_row = rows[1]
    # The +1m envelope omitted t-no, so the horizon midpoint stays null
    # rather than inventing a price.
    assert no_row["baseline_midpoint"] == Decimal("0.60")
    assert no_row["midpoint_plus_1m"] is None
    assert no_row["trade_count"] == 0


def test_reaction_rows_leave_trades_null_when_not_collected() -> None:
    rows = build_reaction_rows(
        link={"news_id": "x:200", "market_id": "m-1", "published_at": NOW},
        tokens=[{"token_id": "t-yes", "outcome_index": 0}],
        envelope_index=[],
        reader=FakeReader({}),
        trade_stats=None,
        computed_at=NOW,
    )
    row = rows[0]
    assert row["trade_count"] is None
    assert row["trade_notional"] is None
    assert row["baseline_midpoint"] is None
    assert row["baseline_spread"] is None
    assert row["baseline_bid_depth"] is None
    assert row["snapshot_count"] == 0


class FakeBlob:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def download_as_bytes(self) -> bytes:
        return self.payload


class FakeBucket:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def blob(self, _path: str) -> FakeBlob:
        return FakeBlob(self.payload)


class FakeStorageClient:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.downloads = 0

    def bucket(self, _name: str) -> FakeBucket:
        self.downloads += 1
        return FakeBucket(self.payload)


def _reader_with_payload(payload: bytes) -> tuple[GcsEnvelopeReader, FakeStorageClient]:
    reader = GcsEnvelopeReader.__new__(GcsEnvelopeReader)
    client = FakeStorageClient(payload)
    reader.client = client  # type: ignore[assignment]
    reader.cache_size = 4
    reader._cache = OrderedDict()
    return reader, client


def test_envelope_reader_handles_gzip_and_transcoded_payloads_with_cache() -> None:
    envelope = {
        "records": [
            {
                "token_id": "t-yes",
                "midpoint": "0.42",
                "spread": "0.04",
                "bid_captured_notional": "812.5",
                "ask_captured_notional": "233",
            }
        ]
    }
    encoded = json.dumps(envelope).encode("utf-8")

    for payload in (gzip.compress(encoded), encoded):
        reader, client = _reader_with_payload(payload)
        books = reader.fetch_books("run-1", "gs://bucket/path.json.gz")
        assert books == {
            "t-yes": {
                "midpoint": "0.42",
                "spread": "0.04",
                "bid_depth": "812.5",
                "ask_depth": "233",
            }
        }
        assert reader.fetch_books("run-1", "gs://bucket/path.json.gz") == books
        assert client.downloads == 1


def test_envelope_reader_keys_kalshi_records_by_ticker() -> None:
    envelope = {
        "records": [
            {
                "ticker": "KXNFLGAME-26AUG15DALSEA-SEA",
                "midpoint": "0.59",
                "spread": "0.02",
                "bid_captured_notional": "5000.00",
                "ask_captured_notional": "6100.00",
            }
        ]
    }
    reader, _client = _reader_with_payload(json.dumps(envelope).encode("utf-8"))
    books = reader.fetch_books("run-k", "gs://bucket/kalshi.json.gz")
    assert books == {
        "KXNFLGAME-26AUG15DALSEA-SEA": {
            "midpoint": "0.59",
            "spread": "0.02",
            "bid_depth": "5000.00",
            "ask_depth": "6100.00",
        }
    }


def test_kalshi_reaction_rows_use_ticker_token_and_platform() -> None:
    ticker = "KXNFLGAME-26AUG15DALSEA-SEA"
    link = {
        "news_id": "x:200",
        "market_id": ticker,
        "platform": "kalshi",
        "published_at": NOW,
    }
    index = [envelope_item(-15, "baseline"), envelope_item(70, "plus-1m")]
    reader = FakeReader(
        {
            "baseline": {ticker: book("0.55")},
            "plus-1m": {ticker: book("0.61")},
        }
    )
    rows = build_reaction_rows(
        link=link,
        tokens=[{"token_id": ticker, "outcome_index": 0}],
        envelope_index=index,
        reader=reader,
        trade_stats={ticker: (4, Decimal("310"))},
        computed_at=NOW + timedelta(hours=3),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["platform"] == "kalshi"
    assert row["token_id"] == ticker
    assert row["outcome_index"] == 0
    assert row["baseline_midpoint"] == Decimal("0.55")
    assert row["delta_plus_1m"] == Decimal("0.06")
    assert row["trade_count"] == 4
    assert row["trade_notional"] == Decimal("310")

    # Polymarket links default the platform.
    default_rows = build_reaction_rows(
        link={"news_id": "x:200", "market_id": "m-1", "published_at": NOW},
        tokens=[{"token_id": "t-yes", "outcome_index": 0}],
        envelope_index=[],
        reader=FakeReader({}),
        trade_stats=None,
        computed_at=NOW,
    )
    assert default_rows[0]["platform"] == "polymarket"


def test_reactions_repository_sql_is_idempotent() -> None:
    connection = FakeConnection()
    resources = SimpleNamespace(engine=FakeEngine(connection), close=lambda: None)
    repository = ReactionsRepository(resources)  # type: ignore[arg-type]

    repository.load_pending_links(cutoff=NOW, limit=10)
    repository.persist_reactions(
        build_reaction_rows(
            link={"news_id": "x:200", "market_id": "m-1", "published_at": NOW},
            tokens=[{"token_id": "t-yes", "outcome_index": 0}],
            envelope_index=[],
            reader=FakeReader({}),
            trade_stats=None,
            computed_at=NOW,
        )
    )

    sql = compiled_sql(connection)
    assert "NOT (EXISTS" in sql
    assert "news_market_links.published_at <=" in sql
    assert "INSERT INTO news_market_reactions" in sql
    assert "ON CONFLICT (news_id, market_id, token_id, label_version) DO UPDATE" in sql


def test_dataset_rows_flatten_values_and_derive_outcome_won() -> None:
    rows = [
        {
            "news_id": "x:200",
            "published_at": NOW,
            "author_username": "Reporter",
            "text": "Player left practice.",
            "market_id": "m-1",
            "event_id": "e-1",
            "question": "Will the player score?",
            "sports_market_type": "props",
            "group_item_title": None,
            "line": Decimal("1.5"),
            "market_topic": "player_props",
            "contract_type": "binary",
            "shared_entity_ids": ["entity-1"],
            "shared_entity_count": 1,
            "news_mention_roles": {"entity-1": {"mention_roles": ["subject"]}},
            "market_mention_roles": {"entity-1": {"mention_roles": ["outcome"]}},
            "market_open_at_publish": True,
            "linker_version": LINKER_VERSION,
            "token_id": "t-yes",
            "outcome": "Yes",
            "outcome_index": 0,
            "label_version": LABEL_VERSION,
            "baseline_midpoint": Decimal("0.40"),
            "baseline_observed_at": NOW - timedelta(seconds=15),
            "baseline_spread": Decimal("0.02"),
            "baseline_bid_depth": Decimal("350"),
            "baseline_ask_depth": Decimal("410.5"),
            "midpoint_plus_1m": Decimal("0.55"),
            "midpoint_plus_5m": None,
            "midpoint_plus_30m": None,
            "midpoint_plus_2h": None,
            "delta_plus_1m": Decimal("0.15"),
            "delta_plus_5m": None,
            "delta_plus_30m": None,
            "delta_plus_2h": None,
            "trade_count": 3,
            "trade_notional": Decimal("120.5"),
            "snapshot_count": 12,
            "max_gap_seconds": Decimal("45"),
            "uma_resolution_status": "resolved",
            "winning_outcome_index": 0,
        }
    ]
    enrichments = {
        "x:200": {
            "summary": "Injury update",
            "information_status": "confirmed",
            "usefulness": "high",
            "claims": [{"claim": "left practice"}],
        }
    }

    dataset = build_dataset_rows(rows, enrichments)

    assert len(dataset) == 1
    row = dataset[0]
    assert row["baseline_midpoint"] == 0.40
    assert row["baseline_spread"] == 0.02
    assert row["baseline_ask_depth"] == 410.5
    assert row["delta_plus_1m"] == 0.15
    assert row["trade_notional"] == 120.5
    assert row["outcome_won"] is True
    assert json.loads(row["claims"]) == [{"claim": "left practice"}]
    assert json.loads(row["shared_entity_ids"]) == ["entity-1"]
    assert row["published_at"] == NOW.isoformat()
    assert row["summary"] == "Injury update"
    assert row["platform"] == "polymarket"
    assert row["market_result"] is None
    assert row["settlement_value"] is None

    dataset_unresolved = build_dataset_rows(
        [{**rows[0], "winning_outcome_index": None}], enrichments
    )
    assert dataset_unresolved[0]["outcome_won"] is None

    kalshi_dataset = build_dataset_rows(
        [
            {
                **rows[0],
                "market_id": "KXNFLGAME-26AUG15DALSEA-SEA",
                "platform": "kalshi",
                "token_id": "KXNFLGAME-26AUG15DALSEA-SEA",
                "market_result": "yes",
                "settlement_value": Decimal("1.0000"),
                "winning_outcome_index": 0,
            }
        ],
        enrichments,
    )
    kalshi_row = kalshi_dataset[0]
    assert kalshi_row["platform"] == "kalshi"
    assert kalshi_row["market_result"] == "yes"
    assert kalshi_row["settlement_value"] == 1.0
    # The reaction row is the yes side, so result "yes" means the outcome won.
    assert kalshi_row["outcome_won"] is True


def label(
    delta: float | None,
    *,
    trade_count: int | None = 5,
    trade_notional: float | None = 250.0,
    spread: float | None = 0.02,
    bid_depth: float | None = 500.0,
    ask_depth: float | None = 500.0,
    rule: BucketRule = BucketRule(),
) -> str | None:
    """bucket_label with a healthy, well-traded book unless overridden."""
    return bucket_label(
        delta,
        trade_count=trade_count,
        trade_notional=trade_notional,
        spread=spread,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        rule=rule,
    )


def test_bucket_label_classifies_confirmed_moves_on_healthy_books() -> None:
    assert label(0.05) == "up"
    assert label(-0.05) == "down"
    assert label(0.005) == "flat"
    # Exactly at the threshold counts as a move.
    assert label(0.02) == "up"
    # No price coverage: unlabeled.
    assert label(None) is None


def test_bucket_label_gates_on_book_quality() -> None:
    # Wide spread: the midpoint is not a price, row is unlabeled.
    assert label(0.05, spread=0.30) is None
    assert label(0.05, spread=None) is None
    # Thin or one-sided book: unlabeled.
    assert label(0.05, bid_depth=20.0) is None
    assert label(0.05, ask_depth=None) is None
    # A move inside half the spread is quote noise, not a move.
    assert label(0.024, spread=0.05) == "flat"
    # Gate off: the same rows label by delta and trades alone.
    ungated = BucketRule(book_quality=False)
    assert label(0.05, spread=0.30, rule=ungated) == "up"
    assert label(0.05, spread=None, bid_depth=None, ask_depth=None, rule=ungated) == "up"


def test_bucket_label_requires_real_executed_volume() -> None:
    # A single tiny trade repricing the book is not a market reaction.
    assert label(0.25, trade_count=1, trade_notional=5.0) == "flat"
    assert label(0.25, trade_count=3, trade_notional=20.0) == "flat"
    assert label(0.25, trade_count=2, trade_notional=80.0) == "up"
    # Trades unknown (window predates collection): unknowable.
    assert label(0.25, trade_count=None, trade_notional=None) is None
    # Small move with unknown trades is still flat (no confirmation needed).
    assert label(0.005, trade_count=None, trade_notional=None) == "flat"
    # With confirmation off, volume is ignored.
    raw = BucketRule(trade_confirmed=False)
    assert label(0.25, trade_count=None, trade_notional=None, rule=raw) == "up"
    assert label(-0.25, trade_count=1, trade_notional=5.0, rule=raw) == "down"


def test_bucket_label_respects_threshold_knob() -> None:
    assert label(0.015, rule=BucketRule(threshold_cents=Decimal("1"))) == "up"
    assert label(0.015, rule=BucketRule(threshold_cents=Decimal("2"))) == "flat"
    assert label(-0.04, rule=BucketRule(threshold_cents=Decimal("5"))) == "flat"


def test_rule_column_names_are_stable() -> None:
    assert rule_column_name(BucketRule()) == "target_plus_30m_2c_tc"
    assert (
        rule_column_name(
            BucketRule(
                horizon="plus_2h",
                threshold_cents=Decimal("5"),
                trade_confirmed=False,
                book_quality=False,
            )
        )
        == "target_plus_2h_5c_raw_nobook"
    )
