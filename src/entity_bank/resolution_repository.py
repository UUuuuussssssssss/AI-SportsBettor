"""Database reads and atomic writes for mention resolution."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import and_, case, exists, or_, select
from sqlalchemy.dialects.postgresql import insert

from src.db.engine import DatabaseResources, create_database_resources
from src.db.models import (
    entities,
    entity_aliases,
    entity_bank_versions,
    entity_mentions,
    entity_relationships,
    entity_resolution_attempts,
    entity_roles,
    kalshi_events,
    kalshi_market_classifications,
    kalshi_markets,
    news_enrichments,
    news_entity_resolution_runs,
    news_events,
    polymarket_events,
    polymarket_market_classifications,
    polymarket_markets,
    polymarket_tokens,
)
from src.entity_bank.models import IdentityStatus


class ResolutionRepository:
    def __init__(self, resources: DatabaseResources) -> None:
        self.resources = resources

    @classmethod
    def from_environment(cls, src_dir: Path) -> ResolutionRepository:
        return cls(create_database_resources(src_dir))

    def close(self) -> None:
        self.resources.close()

    def latest_bank_version_id(self) -> str | None:
        with self.resources.engine.connect() as connection:
            return connection.scalar(
                select(entity_bank_versions.c.version_id)
                .where(entity_bank_versions.c.status == "completed")
                .order_by(entity_bank_versions.c.ingested_at.desc())
                .limit(1)
            )

    def load_market_events(
        self,
        *,
        event_limit: int,
        event_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        statement = (
            select(
                polymarket_events.c.event_id,
                polymarket_events.c.title.label("event_title"),
                polymarket_events.c.slug.label("event_slug"),
                polymarket_events.c.game_id,
                polymarket_markets.c.market_id,
                polymarket_markets.c.question,
                polymarket_markets.c.slug.label("market_slug"),
                polymarket_markets.c.group_item_title,
                polymarket_markets.c.group_item_threshold,
                polymarket_markets.c.sports_market_type,
                polymarket_markets.c.current_content_sha256,
                polymarket_market_classifications.c.entity_input_sha256.label(
                    "prior_entity_input_sha256"
                ),
                polymarket_market_classifications.c.extractor_version.label(
                    "prior_extractor_version"
                ),
                polymarket_tokens.c.outcome,
                polymarket_tokens.c.outcome_index,
            )
            .join(
                polymarket_markets,
                polymarket_markets.c.event_id == polymarket_events.c.event_id,
            )
            .outerjoin(
                polymarket_market_classifications,
                polymarket_market_classifications.c.market_id
                == polymarket_markets.c.market_id,
            )
            .outerjoin(
                polymarket_tokens,
                polymarket_tokens.c.market_id == polymarket_markets.c.market_id,
            )
            .where(
                polymarket_events.c.missing_since.is_(None),
                polymarket_markets.c.missing_since.is_(None),
            )
            .order_by(
                polymarket_events.c.event_id,
                polymarket_markets.c.market_id,
                polymarket_tokens.c.outcome_index,
            )
        )
        if event_ids:
            statement = statement.where(
                polymarket_events.c.event_id.in_(sorted(event_ids))
            )
        with self.resources.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()

        events: dict[str, dict[str, Any]] = {}
        for row in rows:
            event_id = row["event_id"]
            if event_id not in events:
                if len(events) >= event_limit:
                    continue
                events[event_id] = {
                    "event_id": event_id,
                    "title": row["event_title"],
                    "slug": row["event_slug"],
                    "game_id": row["game_id"],
                    "markets": {},
                }
            event = events.get(event_id)
            if event is None:
                continue
            market = event["markets"].setdefault(
                row["market_id"],
                {
                    "market_id": row["market_id"],
                    "question": row["question"],
                    "slug": row["market_slug"],
                    "group_item_title": row["group_item_title"],
                    "group_item_threshold": row["group_item_threshold"],
                    "sports_market_type": row["sports_market_type"],
                    "source_content_sha256": row["current_content_sha256"],
                    "prior_entity_input_sha256": row["prior_entity_input_sha256"],
                    "prior_extractor_version": row["prior_extractor_version"],
                    "outcomes": [],
                },
            )
            if row["outcome"] is not None:
                market["outcomes"].append(row["outcome"])
        return [
            {**event, "markets": list(event["markets"].values())}
            for event in events.values()
        ]

    def load_kalshi_market_events(
        self,
        *,
        event_limit: int,
        event_tickers: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Shape Kalshi events into the ``process_market_events`` input dicts.

        Field mapping: market ``title`` (a full question) -> ``question``,
        ``yes_sub_title`` (the candidate label) -> ``group_item_title``,
        tickers -> slugs, strikes -> ``group_item_threshold``.
        """
        statement = (
            select(
                kalshi_events.c.event_ticker,
                kalshi_events.c.title.label("event_title"),
                kalshi_markets.c.ticker,
                kalshi_markets.c.title.label("market_title"),
                kalshi_markets.c.yes_sub_title,
                kalshi_markets.c.no_sub_title,
                kalshi_markets.c.floor_strike,
                kalshi_markets.c.cap_strike,
                kalshi_markets.c.functional_strike,
                kalshi_markets.c.current_content_sha256,
                kalshi_market_classifications.c.entity_input_sha256.label(
                    "prior_entity_input_sha256"
                ),
                kalshi_market_classifications.c.extractor_version.label(
                    "prior_extractor_version"
                ),
            )
            .join(
                kalshi_markets,
                kalshi_markets.c.event_ticker == kalshi_events.c.event_ticker,
            )
            .outerjoin(
                kalshi_market_classifications,
                kalshi_market_classifications.c.market_ticker
                == kalshi_markets.c.ticker,
            )
            .where(
                kalshi_events.c.missing_since.is_(None),
                kalshi_markets.c.missing_since.is_(None),
            )
            .order_by(kalshi_events.c.event_ticker, kalshi_markets.c.ticker)
        )
        if event_tickers:
            statement = statement.where(
                kalshi_events.c.event_ticker.in_(sorted(event_tickers))
            )
        with self.resources.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()

        events: dict[str, dict[str, Any]] = {}
        for row in rows:
            event_ticker = row["event_ticker"]
            if event_ticker not in events:
                if len(events) >= event_limit:
                    continue
                events[event_ticker] = {
                    "event_id": event_ticker,
                    "title": row["event_title"],
                    "slug": event_ticker,
                    "markets": [],
                }
            event = events.get(event_ticker)
            if event is None:
                continue
            if row["functional_strike"]:
                threshold = str(row["functional_strike"])
            else:
                strikes = [
                    str(value)
                    for value in (row["floor_strike"], row["cap_strike"])
                    if value is not None
                ]
                threshold = "-".join(strikes) or None
            outcomes: list[str] = []
            for sub_title in (row["yes_sub_title"], row["no_sub_title"]):
                if sub_title and sub_title not in outcomes:
                    outcomes.append(sub_title)
            event["markets"].append(
                {
                    "market_id": row["ticker"],
                    "question": row["market_title"],
                    "slug": row["ticker"],
                    "group_item_title": row["yes_sub_title"],
                    "group_item_threshold": threshold,
                    "sports_market_type": None,
                    "source_content_sha256": row["current_content_sha256"],
                    "prior_entity_input_sha256": row["prior_entity_input_sha256"],
                    "prior_extractor_version": row["prior_extractor_version"],
                    "outcomes": outcomes,
                }
            )
        return list(events.values())

    def load_news(
        self,
        *,
        limit: int,
        extractor_version: str,
        enrichment_version: str,
        news_id: str | None = None,
        input_fingerprint: str | None = None,
    ) -> list[dict[str, Any]]:
        already_processed = exists(
            select(news_entity_resolution_runs.c.news_id).where(
                news_entity_resolution_runs.c.news_id == news_enrichments.c.news_id,
                news_entity_resolution_runs.c.enrichment_version
                == news_enrichments.c.enrichment_version,
                news_entity_resolution_runs.c.input_fingerprint
                == news_enrichments.c.input_fingerprint,
                news_entity_resolution_runs.c.extractor_version == extractor_version,
                news_entity_resolution_runs.c.status == "completed",
            )
        )
        statement = (
            select(
                news_events.c.news_id,
                news_events.c.text,
                news_enrichments.c.input_fingerprint,
                news_enrichments.c.enrichment_version,
                news_enrichments.c.entities,
                news_enrichments.c.summary,
                news_enrichments.c.completed_at,
            )
            .join(news_enrichments, news_enrichments.c.news_id == news_events.c.news_id)
            .where(
                news_enrichments.c.status.like("completed%"),
                news_enrichments.c.enrichment_version == enrichment_version,
                news_enrichments.c.entity_extractor_version == extractor_version,
                ~already_processed,
            )
            .order_by(news_enrichments.c.completed_at)
            .limit(limit)
        )
        if news_id is not None:
            statement = statement.where(news_events.c.news_id == news_id)
        if input_fingerprint is not None:
            statement = statement.where(
                news_enrichments.c.input_fingerprint == input_fingerprint
            )
        with self.resources.engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement).mappings()]

    def load_candidate_rows(self) -> list[dict[str, Any]]:
        latest_completed_versions = (
            select(entity_bank_versions.c.version_id)
            .where(
                entity_bank_versions.c.status == "completed",
            )
            .distinct(entity_bank_versions.c.source)
            .order_by(
                entity_bank_versions.c.source,
                entity_bank_versions.c.ingested_at.desc(),
            )
        )
        alias_statement = (
            select(
                entities.c.entity_id,
                entities.c.canonical_name,
                entities.c.entity_type,
                entities.c.identity_status,
                entity_aliases.c.alias,
            )
            .join(entity_aliases, entity_aliases.c.entity_id == entities.c.entity_id)
            .where(
                or_(
                    entities.c.identity_status == IdentityStatus.PROVISIONAL.value,
                    and_(
                        entities.c.identity_status == IdentityStatus.CANONICAL.value,
                        entities.c.latest_bank_version_id.in_(latest_completed_versions),
                    ),
                )
            )
            .order_by(entities.c.entity_id)
        )
        role_statement = select(entity_roles.c.entity_id, entity_roles.c.role).where(
            entity_roles.c.valid_to.is_(None)
        )
        team = entities.alias("team")
        team_statement = (
            select(
                entity_relationships.c.subject_entity_id.label("entity_id"),
                team.c.canonical_name.label("team_name"),
            )
            .join(team, team.c.entity_id == entity_relationships.c.object_entity_id)
            .where(
                entity_relationships.c.predicate.in_(("rostered_by", "coaches_for")),
                or_(
                    entity_relationships.c.valid_to.is_(None),
                    entity_relationships.c.valid_to >= datetime.now(UTC),
                ),
            )
        )
        with self.resources.engine.connect() as connection:
            alias_rows = connection.execute(alias_statement).mappings().all()
            role_rows = connection.execute(role_statement).mappings().all()
            team_rows = connection.execute(team_statement).mappings().all()

        aggregated: dict[str, dict[str, Any]] = {}
        for row in alias_rows:
            aggregated.setdefault(
                row["entity_id"],
                {
                    "entity_id": row["entity_id"],
                    "canonical_name": row["canonical_name"],
                    "entity_type": row["entity_type"],
                    "identity_status": row["identity_status"],
                    "aliases": [],
                    "roles": [],
                    "teams": [],
                },
            )["aliases"].append(row["alias"])
        for row in role_rows:
            if row["entity_id"] in aggregated:
                aggregated[row["entity_id"]]["roles"].append(row["role"])
        for row in team_rows:
            if row["entity_id"] in aggregated:
                aggregated[row["entity_id"]]["teams"].append(row["team_name"])
        return list(aggregated.values())

    def load_pending_mentions(
        self,
        *,
        bank_version_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        statement = (
            select(entity_mentions)
            .where(
                entity_mentions.c.resolution_status.in_(["ambiguous", "unresolved"]),
                entity_mentions.c.last_bank_version_id.is_distinct_from(bank_version_id),
                entity_mentions.c.resolution_metadata[
                    "manual_lock"
                ].as_boolean().is_not(True),
            )
            .order_by(entity_mentions.c.updated_at)
            .limit(limit)
        )
        with self.resources.engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement).mappings()]

    def load_mentions_for_accuracy_sweep(
        self,
        *,
        scope: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Load bounded source context for a no-write, manually triggered audit."""

        statuses = {
            "needs_review": ["ambiguous", "unresolved"],
            "resolved": ["resolved"],
            "all": ["ambiguous", "unresolved", "resolved", "ignored"],
        }.get(scope)
        if statuses is None:
            raise ValueError("scope must be needs_review, resolved, or all")

        market_event = polymarket_events.alias("sweep_market_event")
        direct_event = polymarket_events.alias("sweep_direct_event")
        statement = (
            select(
                entity_mentions,
                news_events.c.text.label("news_text"),
                polymarket_markets.c.question.label("market_question"),
                polymarket_markets.c.slug.label("market_slug"),
                market_event.c.title.label("market_event_title"),
                direct_event.c.title.label("direct_event_title"),
            )
            .outerjoin(news_events, news_events.c.news_id == entity_mentions.c.news_id)
            .outerjoin(
                polymarket_markets,
                polymarket_markets.c.market_id
                == entity_mentions.c.polymarket_market_id,
            )
            .outerjoin(
                market_event,
                market_event.c.event_id == polymarket_markets.c.event_id,
            )
            .outerjoin(
                direct_event,
                direct_event.c.event_id == entity_mentions.c.polymarket_event_id,
            )
            .where(
                entity_mentions.c.resolution_status.in_(statuses),
                entity_mentions.c.resolution_metadata[
                    "manual_lock"
                ].as_boolean().is_not(True),
            )
            .order_by(
                case(
                    (entity_mentions.c.resolution_status == "ambiguous", 0),
                    (entity_mentions.c.resolution_status == "unresolved", 1),
                    (entity_mentions.c.resolution_status == "resolved", 2),
                    else_=3,
                ),
                entity_mentions.c.updated_at,
                entity_mentions.c.mention_id,
            )
            .limit(limit)
        )
        with self.resources.engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement).mappings()]

    def persist_batch(self, batch: dict[str, list[dict[str, Any]]]) -> None:
        now = datetime.now(UTC)
        with self.resources.engine.begin() as connection:
            provisional_rows = batch.get("provisional_entities", [])
            if provisional_rows:
                statement = insert(entities)
                connection.execute(
                    statement.on_conflict_do_nothing(index_elements=[entities.c.entity_id]),
                    provisional_rows,
                )
            alias_rows = batch.get("aliases", [])
            if alias_rows:
                statement = insert(entity_aliases)
                connection.execute(
                    statement.on_conflict_do_update(
                        index_elements=[
                            entity_aliases.c.entity_id,
                            entity_aliases.c.normalized_alias,
                            entity_aliases.c.source,
                        ],
                        set_={
                            "last_observed_at": statement.excluded.last_observed_at,
                            "alias": statement.excluded.alias,
                        },
                    ),
                    alias_rows,
                )
            classification_rows = batch.get("classifications", [])
            # Kalshi rows are keyed by market_ticker, Polymarket rows by
            # market_id; route each to its own table.
            polymarket_rows = [
                row for row in classification_rows if "market_ticker" not in row
            ]
            kalshi_rows = [
                row for row in classification_rows if "market_ticker" in row
            ]
            if polymarket_rows:
                statement = insert(polymarket_market_classifications)
                connection.execute(
                    statement.on_conflict_do_update(
                        index_elements=[polymarket_market_classifications.c.market_id],
                        set_={
                            column.name: getattr(statement.excluded, column.name)
                            for column in (
                                polymarket_market_classifications.c.source_content_sha256,
                                polymarket_market_classifications.c.entity_input_sha256,
                                polymarket_market_classifications.c.market_topic,
                                polymarket_market_classifications.c.contract_type,
                                polymarket_market_classifications.c.extractor_version,
                                polymarket_market_classifications.c.confidence,
                                polymarket_market_classifications.c.classification_metadata,
                                polymarket_market_classifications.c.updated_at,
                            )
                        },
                    ),
                    polymarket_rows,
                )
            if kalshi_rows:
                statement = insert(kalshi_market_classifications)
                connection.execute(
                    statement.on_conflict_do_update(
                        index_elements=[kalshi_market_classifications.c.market_ticker],
                        set_={
                            column.name: getattr(statement.excluded, column.name)
                            for column in (
                                kalshi_market_classifications.c.source_content_sha256,
                                kalshi_market_classifications.c.entity_input_sha256,
                                kalshi_market_classifications.c.market_topic,
                                kalshi_market_classifications.c.contract_type,
                                kalshi_market_classifications.c.extractor_version,
                                kalshi_market_classifications.c.confidence,
                                kalshi_market_classifications.c.classification_metadata,
                                kalshi_market_classifications.c.updated_at,
                            )
                        },
                    ),
                    kalshi_rows,
                )
            # Retried mentions are full table rows (including created_at) while
            # freshly extracted ones are not; executemany requires homogeneous
            # keys, and created_at belongs to the insert default anyway.
            mention_rows = [
                {key: value for key, value in row.items() if key != "created_at"}
                for row in batch.get("mentions", [])
            ]
            if mention_rows:
                statement = insert(entity_mentions)
                connection.execute(
                    statement.on_conflict_do_update(
                        index_elements=[entity_mentions.c.mention_id],
                        set_={
                            "entity_id": statement.excluded.entity_id,
                            "resolver_version": statement.excluded.resolver_version,
                            "resolution_status": statement.excluded.resolution_status,
                            "match_method": statement.excluded.match_method,
                            "confidence": statement.excluded.confidence,
                            "last_bank_version_id": statement.excluded.last_bank_version_id,
                            "candidate_entity_ids": (
                                statement.excluded.candidate_entity_ids
                            ),
                            "resolution_metadata": statement.excluded.resolution_metadata,
                            "last_observed_at": statement.excluded.last_observed_at,
                            "updated_at": now,
                        },
                        where=entity_mentions.c.resolution_metadata[
                            "manual_lock"
                        ].as_boolean().is_not(True),
                    ),
                    mention_rows,
                )
            attempt_rows = batch.get("attempts", [])
            if attempt_rows:
                statement = insert(entity_resolution_attempts)
                connection.execute(
                    statement.on_conflict_do_nothing(
                        index_elements=[entity_resolution_attempts.c.attempt_id]
                    ),
                    attempt_rows,
                )
            run_rows = batch.get("news_resolution_runs", [])
            if run_rows:
                statement = insert(news_entity_resolution_runs)
                connection.execute(
                    statement.on_conflict_do_update(
                        index_elements=[
                            news_entity_resolution_runs.c.news_id,
                            news_entity_resolution_runs.c.enrichment_version,
                            news_entity_resolution_runs.c.input_fingerprint,
                            news_entity_resolution_runs.c.extractor_version,
                        ],
                        set_={
                            "bank_version_id": statement.excluded.bank_version_id,
                            "status": statement.excluded.status,
                            "mention_count": statement.excluded.mention_count,
                            "failure_count": statement.excluded.failure_count,
                            "completed_at": statement.excluded.completed_at,
                            "updated_at": statement.excluded.updated_at,
                        },
                    ),
                    run_rows,
                )
