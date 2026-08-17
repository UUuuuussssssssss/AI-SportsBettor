"""Deterministic tweet-market linking through shared resolved entities.

Every training example needs a (tweet, market) pair. This linker joins
resolved news mentions to resolved market mentions on canonical entity IDs
and records quality features (shared-entity count, mention roles, market
topic). A tweet is linked only to markets that were already observed and
still tradable at publish time: a later listing cannot have a price
reaction, and a market that was already final cannot either.

Hourly runs are incremental. Each tweet is linked once, against the open
book as it existed then. New markets do not walk back through old tweets.
`--full` still rebuilds the whole graph for an explicit backfill.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import delete, exists, or_, select
from sqlalchemy.dialects.postgresql import insert

from src.db.engine import DatabaseResources, create_database_resources
from src.db.models import (
    entity_mentions,
    ingest_cursors,
    kalshi_market_classifications,
    kalshi_markets,
    news_entity_resolution_runs,
    news_events,
    news_market_links,
    polymarket_market_classifications,
    polymarket_markets,
)

LINKER_VERSION = "entity_overlap_v3"
LINKER_CURSOR_SOURCE = "linking"
LINKER_CURSOR_STREAM = "news_market_links"
RESOLVED_STATUS = "resolved"
INCREMENTAL_LOOKBACK = timedelta(hours=24)


def utc_now() -> datetime:
    return datetime.now(UTC)


def linker_fingerprint() -> str:
    return hashlib.sha256(LINKER_VERSION.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NewsMention:
    news_id: str
    entity_id: str
    mention_role: str
    person_role_hint: str | None
    published_at: datetime


@dataclass(frozen=True)
class MarketMention:
    market_id: str
    entity_id: str
    mention_role: str
    person_role_hint: str | None
    event_id: str
    closed_time: datetime | None
    resolution_observed_at: datetime | None
    first_observed_at: datetime | None
    market_topic: str | None
    contract_type: str | None
    platform: str = "polymarket"


def market_final_before(mention: MarketMention, published_at: datetime) -> bool:
    """True when the market was provably final before the tweet published."""
    if mention.closed_time is not None and mention.closed_time < published_at:
        return True
    return (
        mention.resolution_observed_at is not None
        and mention.resolution_observed_at < published_at
    )


def market_open_at_publish(mention: MarketMention, published_at: datetime) -> bool:
    """True when we had already observed the market before the tweet published."""
    if mention.first_observed_at is None:
        return False
    return mention.first_observed_at <= published_at


def _role_map(pairs: set[tuple[str, str, str | None]]) -> dict[str, dict[str, list[str]]]:
    roles: dict[str, set[str]] = defaultdict(set)
    hints: dict[str, set[str]] = defaultdict(set)
    for entity_id, mention_role, person_role_hint in pairs:
        roles[entity_id].add(mention_role)
        if person_role_hint:
            hints[entity_id].add(person_role_hint)
    return {
        entity_id: {
            "mention_roles": sorted(roles[entity_id]),
            "person_role_hints": sorted(hints.get(entity_id, set())),
        }
        for entity_id in roles
    }


def build_links(
    news_mentions: list[NewsMention],
    market_mentions: list[MarketMention],
) -> list[dict[str, Any]]:
    """Join the two mention sides on entity_id and aggregate per pair."""
    markets_by_entity: dict[str, list[MarketMention]] = defaultdict(list)
    for market_mention in market_mentions:
        markets_by_entity[market_mention.entity_id].append(market_mention)

    news_roles: dict[tuple[str, str], set[tuple[str, str, str | None]]] = defaultdict(set)
    market_roles: dict[tuple[str, str], set[tuple[str, str, str | None]]] = defaultdict(set)
    pair_market: dict[tuple[str, str], MarketMention] = {}
    pair_published: dict[tuple[str, str], datetime] = {}

    for news_mention in news_mentions:
        for market_mention in markets_by_entity.get(news_mention.entity_id, ()):
            if market_final_before(market_mention, news_mention.published_at):
                continue
            if not market_open_at_publish(market_mention, news_mention.published_at):
                continue
            pair = (news_mention.news_id, market_mention.market_id)
            pair_market.setdefault(pair, market_mention)
            pair_published.setdefault(pair, news_mention.published_at)
            news_roles[pair].add(
                (
                    news_mention.entity_id,
                    news_mention.mention_role,
                    news_mention.person_role_hint,
                )
            )
            market_roles[pair].add(
                (
                    market_mention.entity_id,
                    market_mention.mention_role,
                    market_mention.person_role_hint,
                )
            )

    rows: list[dict[str, Any]] = []
    for pair in sorted(pair_market):
        news_id, market_id = pair
        market_mention = pair_market[pair]
        published_at = pair_published[pair]
        shared_entity_ids = sorted(
            {entity_id for entity_id, _, _ in news_roles[pair]}
        )
        rows.append(
            {
                "news_id": news_id,
                "market_id": market_id,
                "event_id": market_mention.event_id,
                "platform": market_mention.platform,
                "published_at": published_at,
                "shared_entity_ids": shared_entity_ids,
                "shared_entity_count": len(shared_entity_ids),
                "news_mention_roles": _role_map(news_roles[pair]),
                "market_mention_roles": _role_map(market_roles[pair]),
                "market_topic": market_mention.market_topic,
                "contract_type": market_mention.contract_type,
                "market_open_at_publish": True,
                "linker_version": LINKER_VERSION,
            }
        )
    return rows


class LinkerRepository:
    def __init__(self, resources: DatabaseResources) -> None:
        self.resources = resources

    @classmethod
    def from_environment(cls, src_dir: Path) -> LinkerRepository:
        return cls(create_database_resources(src_dir))

    def close(self) -> None:
        self.resources.close()

    def last_successful_run_at(self) -> datetime | None:
        statement = select(ingest_cursors.c.last_successful_poll_at).where(
            ingest_cursors.c.source == LINKER_CURSOR_SOURCE,
            ingest_cursors.c.stream == LINKER_CURSOR_STREAM,
        )
        with self.resources.engine.connect() as connection:
            value = connection.scalar(statement)
        return value

    def load_news_mentions(
        self,
        *,
        since: datetime | None = None,
        skip_linked: bool = True,
    ) -> list[NewsMention]:
        already_linked = exists(
            select(news_market_links.c.news_id).where(
                news_market_links.c.news_id == entity_mentions.c.news_id
            )
        )
        statement = (
            select(
                entity_mentions.c.news_id,
                entity_mentions.c.entity_id,
                entity_mentions.c.mention_role,
                entity_mentions.c.person_role_hint,
                news_events.c.published_at,
            )
            .join(news_events, news_events.c.news_id == entity_mentions.c.news_id)
            .where(
                entity_mentions.c.news_id.is_not(None),
                entity_mentions.c.resolution_status == RESOLVED_STATUS,
                entity_mentions.c.entity_id.is_not(None),
            )
            .order_by(entity_mentions.c.news_id)
        )
        if skip_linked:
            statement = statement.where(~already_linked)
        if since is not None:
            lookback = since - INCREMENTAL_LOOKBACK
            recently_resolved = exists(
                select(news_entity_resolution_runs.c.news_id).where(
                    news_entity_resolution_runs.c.news_id == entity_mentions.c.news_id,
                    news_entity_resolution_runs.c.status == "completed",
                    news_entity_resolution_runs.c.completed_at > since,
                )
            )
            statement = statement.where(
                or_(
                    news_events.c.published_at > lookback,
                    recently_resolved,
                )
            )
        with self.resources.engine.connect() as connection:
            return [
                NewsMention(
                    news_id=row.news_id,
                    entity_id=row.entity_id,
                    mention_role=row.mention_role,
                    person_role_hint=row.person_role_hint,
                    published_at=row.published_at,
                )
                for row in connection.execute(statement)
            ]

    def load_market_mentions(self) -> list[MarketMention]:
        return [
            *self._load_polymarket_market_mentions(),
            *self._load_kalshi_market_mentions(),
        ]

    def _load_polymarket_market_mentions(self) -> list[MarketMention]:
        statement = (
            select(
                entity_mentions.c.polymarket_market_id,
                entity_mentions.c.entity_id,
                entity_mentions.c.mention_role,
                entity_mentions.c.person_role_hint,
                polymarket_markets.c.event_id,
                polymarket_markets.c.closed_time,
                polymarket_markets.c.resolution_observed_at,
                polymarket_markets.c.first_observed_at,
                polymarket_market_classifications.c.market_topic,
                polymarket_market_classifications.c.contract_type,
            )
            .join(
                polymarket_markets,
                polymarket_markets.c.market_id == entity_mentions.c.polymarket_market_id,
            )
            .join(
                polymarket_market_classifications,
                polymarket_market_classifications.c.market_id
                == polymarket_markets.c.market_id,
                isouter=True,
            )
            .where(
                entity_mentions.c.polymarket_market_id.is_not(None),
                entity_mentions.c.resolution_status == RESOLVED_STATUS,
                entity_mentions.c.entity_id.is_not(None),
            )
            .order_by(entity_mentions.c.polymarket_market_id)
        )
        with self.resources.engine.connect() as connection:
            return [
                MarketMention(
                    market_id=row.polymarket_market_id,
                    entity_id=row.entity_id,
                    mention_role=row.mention_role,
                    person_role_hint=row.person_role_hint,
                    event_id=row.event_id,
                    closed_time=row.closed_time,
                    resolution_observed_at=row.resolution_observed_at,
                    first_observed_at=row.first_observed_at,
                    market_topic=row.market_topic,
                    contract_type=row.contract_type,
                    platform="polymarket",
                )
                for row in connection.execute(statement)
            ]

    def _load_kalshi_market_mentions(self) -> list[MarketMention]:
        # Kalshi analogs: close_time is the scheduled end of trading and
        # settlement_ts is the actual settlement, so either one preceding the
        # tweet makes the market final.
        statement = (
            select(
                entity_mentions.c.kalshi_market_ticker,
                entity_mentions.c.entity_id,
                entity_mentions.c.mention_role,
                entity_mentions.c.person_role_hint,
                kalshi_markets.c.event_ticker,
                kalshi_markets.c.close_time,
                kalshi_markets.c.settlement_ts,
                kalshi_markets.c.first_observed_at,
                kalshi_market_classifications.c.market_topic,
                kalshi_market_classifications.c.contract_type,
            )
            .join(
                kalshi_markets,
                kalshi_markets.c.ticker == entity_mentions.c.kalshi_market_ticker,
            )
            .join(
                kalshi_market_classifications,
                kalshi_market_classifications.c.market_ticker
                == kalshi_markets.c.ticker,
                isouter=True,
            )
            .where(
                entity_mentions.c.kalshi_market_ticker.is_not(None),
                entity_mentions.c.resolution_status == RESOLVED_STATUS,
                entity_mentions.c.entity_id.is_not(None),
            )
            .order_by(entity_mentions.c.kalshi_market_ticker)
        )
        with self.resources.engine.connect() as connection:
            return [
                MarketMention(
                    market_id=row.kalshi_market_ticker,
                    entity_id=row.entity_id,
                    mention_role=row.mention_role,
                    person_role_hint=row.person_role_hint,
                    event_id=row.event_ticker,
                    closed_time=row.close_time,
                    resolution_observed_at=row.settlement_ts,
                    first_observed_at=row.first_observed_at,
                    market_topic=row.market_topic,
                    contract_type=row.contract_type,
                    platform="kalshi",
                )
                for row in connection.execute(statement)
            ]

    def delete_unopened_links(self) -> int:
        """Drop pairs whose market was first observed after the tweet."""
        statement = delete(news_market_links).where(
            news_market_links.c.market_open_at_publish.is_(False)
        )
        with self.resources.engine.begin() as connection:
            result = connection.execute(statement)
        return int(result.rowcount or 0)

    def persist_links(
        self,
        rows: list[dict[str, Any]],
        *,
        run_started_at: datetime,
        prune: bool = False,
    ) -> int:
        """Upsert computed links. Full rebuilds also prune rows not regenerated."""
        if not rows and not prune:
            return 0
        with self.resources.engine.begin() as connection:
            for offset in range(0, len(rows), 500):
                batch = [
                    {**row, "updated_at": run_started_at}
                    for row in rows[offset : offset + 500]
                ]
                statement = insert(news_market_links).values(batch)
                connection.execute(
                    statement.on_conflict_do_update(
                        index_elements=[
                            news_market_links.c.news_id,
                            news_market_links.c.market_id,
                        ],
                        set_={
                            column: getattr(statement.excluded, column)
                            for column in (
                                "event_id",
                                "platform",
                                "published_at",
                                "shared_entity_ids",
                                "shared_entity_count",
                                "news_mention_roles",
                                "market_mention_roles",
                                "market_topic",
                                "contract_type",
                                "market_open_at_publish",
                                "linker_version",
                                "updated_at",
                            )
                        },
                    )
                )
            pruned = 0
            if prune:
                result = connection.execute(
                    news_market_links.delete().where(
                        news_market_links.c.updated_at < run_started_at
                    )
                )
                pruned = result.rowcount or 0
        return pruned

    def finalize_run(self, *, run_started_at: datetime) -> None:
        cursor_insert = insert(ingest_cursors).values(
            source=LINKER_CURSOR_SOURCE,
            stream=LINKER_CURSOR_STREAM,
            query_fingerprint=linker_fingerprint(),
            last_structural_sha256=None,
            since_id=None,
            updated_at=run_started_at,
            last_successful_poll_at=run_started_at,
        )
        with self.resources.engine.begin() as connection:
            connection.execute(
                cursor_insert.on_conflict_do_update(
                    index_elements=[
                        ingest_cursors.c.source,
                        ingest_cursors.c.stream,
                    ],
                    set_={
                        "query_fingerprint": cursor_insert.excluded.query_fingerprint,
                        "updated_at": cursor_insert.excluded.updated_at,
                        "last_successful_poll_at": (
                            cursor_insert.excluded.last_successful_poll_at
                        ),
                    },
                    where=(
                        cursor_insert.excluded.last_successful_poll_at
                        >= ingest_cursors.c.last_successful_poll_at
                    ),
                )
            )


def run_linker(
    repository: LinkerRepository,
    *,
    dry_run: bool = False,
    full: bool = False,
) -> list[dict[str, Any]]:
    run_started_at = utc_now()
    since = None if full else repository.last_successful_run_at()
    news_mentions = repository.load_news_mentions(
        since=since,
        skip_linked=not full,
    )
    market_mentions = repository.load_market_mentions()
    mode = "full" if full else "incremental"
    window = (
        "all tweets"
        if since is None or full
        else f"unlinked since {since.isoformat()}"
    )
    print(
        f"Loaded {len(news_mentions)} resolved news mentions and "
        f"{len(market_mentions)} resolved market mentions ({mode}, {window})"
    )
    rows = build_links(news_mentions, market_mentions)
    linked_news = len({row["news_id"] for row in rows})
    linked_markets = len({row["market_id"] for row in rows})
    print(
        f"Computed {len(rows)} tweet-market links covering "
        f"{linked_news} tweets and {linked_markets} markets"
    )
    if dry_run:
        print("DRY RUN: skipped PostgreSQL writes")
        return rows
    dropped = repository.delete_unopened_links()
    if dropped:
        print(f"Removed {dropped} links whose markets opened after the tweet")
    pruned = repository.persist_links(
        rows,
        run_started_at=run_started_at,
        prune=full,
    )
    repository.finalize_run(run_started_at=run_started_at)
    print(
        f"Committed {len(rows)} links (pruned {pruned} stale) at {LINKER_VERSION}"
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report links without writing to PostgreSQL",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Rebuild every tweet against the current book and prune stale rows",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    src_dir = Path(__file__).resolve().parents[1]
    repository: LinkerRepository | None = None
    try:
        repository = LinkerRepository.from_environment(src_dir)
        run_linker(repository, dry_run=args.dry_run, full=args.full)
        return 0
    except Exception as exc:
        print(f"ERROR: tweet-market linking failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if repository:
            repository.close()
