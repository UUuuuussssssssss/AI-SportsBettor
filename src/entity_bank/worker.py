"""Resolve Polymarket/X entities; local JSONL dry run unless --apply is explicit."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.enrich_news.config import load_enrichment_settings
from src.enrich_news.models import EntityType as EnrichmentEntityType
from src.enrich_news.prompt import ENTITY_EXTRACTOR_VERSION
from src.entity_bank.models import (
    ContractType,
    EntityType,
    ExtractedMention,
    MarketTopic,
    MentionRole,
    PersonRoleHint,
)
from src.entity_bank.normalization import (
    entity_input_fingerprint,
    normalize_name,
    placeholder_reason,
)
from src.entity_bank.prompt import EXTRACTOR_VERSION, RESOLVER_VERSION
from src.entity_bank.provider import (
    ClaudeEntityProvider,
    DeterministicEntityProvider,
    EntityProvider,
)
from src.entity_bank.resolution_repository import ResolutionRepository
from src.entity_bank.resolver import (
    CandidateIndex,
    SourceReference,
    deterministic_alias_match_is_safe,
    ignored_mention,
    resolve_mention,
    serialize_audit_record,
)

WRITE_CONFIRMATION = "APPLY_ENTITY_RESOLUTIONS"
MARKET_CHUNK_SIZE = 20
PROVISIONAL_PERSON_TOPICS = {
    MarketTopic.GAME,
    MarketTopic.CHAMPIONSHIP,
    MarketTopic.POSTSEASON_QUALIFICATION,
    MarketTopic.AWARD,
    MarketTopic.STAT_LEADER,
    MarketTopic.DRAFT,
    MarketTopic.ROSTER_DESTINATION,
    MarketTopic.STARTING_ROLE,
    MarketTopic.TRANSACTION,
    MarketTopic.RETIREMENT,
    MarketTopic.COACHING_STATUS,
    MarketTopic.INJURY_AVAILABILITY,
    MarketTopic.CONTRACT,
}
PROVISIONAL_PERSON_ROLES = {
    PersonRoleHint.PLAYER,
    PersonRoleHint.COACH,
    PersonRoleHint.PROSPECT,
    PersonRoleHint.OWNER,
    PersonRoleHint.EXECUTIVE,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("markets", "news", "both"), default="markets")
    parser.add_argument("--event-limit", type=int, default=10)
    parser.add_argument("--news-limit", type=int, default=20)
    parser.add_argument("--retry-limit", type=int, default=100)
    parser.add_argument("--provider", choices=("mock", "claude"), default="claude")
    parser.add_argument("--model", help="Override NEWS_ENRICHMENT_MODEL")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist entity/classification/resolution rows to PostgreSQL",
    )
    parser.add_argument(
        "--confirm-live-writes",
        help=f"Required with --apply; must equal {WRITE_CONFIRMATION}",
    )
    return parser


def _chunks(values: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


class Batch:
    def __init__(self) -> None:
        self.classifications: dict[str, dict[str, Any]] = {}
        self.mentions: dict[str, dict[str, Any]] = {}
        self.attempts: dict[str, dict[str, Any]] = {}
        self.provisional_entities: dict[str, dict[str, Any]] = {}
        self.aliases: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.news_resolution_runs: dict[
            tuple[str, str, str, str],
            dict[str, Any],
        ] = {}
        self.failures: list[dict[str, Any]] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.skipped_unchanged_markets = 0
        self.skipped_unchanged_candidate_sets = 0

    def add_resolution(self, result: dict[str, Any]) -> None:
        mention = result["mention"]
        attempt = result["attempt"]
        self.mentions[mention["mention_id"]] = mention
        self.attempts[attempt["attempt_id"]] = attempt
        if result["provisional_entity"]:
            entity = result["provisional_entity"]
            self.provisional_entities[entity["entity_id"]] = entity
        if result["provisional_alias"]:
            alias = result["provisional_alias"]
            key = (alias["entity_id"], alias["normalized_alias"], alias["source"])
            self.aliases[key] = alias
        self.input_tokens += result["provider_usage"]["input_tokens"]
        self.output_tokens += result["provider_usage"]["output_tokens"]

    def as_repository_batch(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "classifications": list(self.classifications.values()),
            "mentions": list(self.mentions.values()),
            "attempts": list(self.attempts.values()),
            "provisional_entities": list(self.provisional_entities.values()),
            "aliases": list(self.aliases.values()),
            "news_resolution_runs": list(self.news_resolution_runs.values()),
        }


def _validate_extracted_evidence(
    mention: ExtractedMention,
    *,
    source_context: str,
    source_fields: dict[str, str] | None = None,
) -> None:
    normalized_text = normalize_name(mention.text)
    normalized_evidence = normalize_name(mention.evidence)
    if normalized_text not in normalize_name(source_context):
        raise ValueError(f"extracted mention absent from source evidence: {mention.text!r}")
    if normalized_evidence not in normalize_name(source_context):
        raise ValueError(f"mention evidence is not verbatim: {mention.evidence!r}")
    if normalized_text not in normalized_evidence:
        raise ValueError(
            f"mention evidence does not identify its entity: "
            f"{mention.text!r} not in {mention.evidence!r}"
        )
    if source_fields is not None:
        unknown_refs = sorted(set(mention.source_refs) - set(source_fields))
        if unknown_refs:
            raise ValueError(f"unknown market source references: {unknown_refs}")
        supporting_refs = {
            name
            for name, value in source_fields.items()
            if normalized_evidence in normalize_name(value)
        }
        if not mention.source_refs or not set(mention.source_refs).intersection(
            supporting_refs
        ):
            raise ValueError(
                f"market source references do not support evidence: "
                f"{mention.source_refs!r}"
            )


def _analyze_market_chunk(
    provider: EntityProvider,
    *,
    event: dict[str, Any],
    markets: list[dict[str, Any]],
) -> list[Any]:
    try:
        result = provider.analyze_market_event(
            event_id=event["event_id"],
            event_title=event["title"],
            event_slug=event["slug"],
            markets=markets,
        )
        return [result]
    except Exception:
        if len(markets) == 1:
            raise
        results = []
        for market in markets:
            results.extend(
                _analyze_market_chunk(provider, event=event, markets=[market])
            )
        return results


def process_market_events(
    *,
    events: list[dict[str, Any]],
    provider: EntityProvider,
    index: CandidateIndex,
    bank_version_id: str | None,
    batch: Batch,
    observed_at: datetime,
    source_kind: str = "polymarket_market",
) -> None:
    """Classify markets and resolve their entity mentions.

    ``source_kind`` selects where mentions and classifications land:
    ``polymarket_market`` (Gamma ids) or ``kalshi_market`` (tickers). Loaders
    shape both platforms into the same event/market dicts, so everything else
    is shared.
    """
    is_kalshi = source_kind == "kalshi_market"
    event_label = "kalshi_event" if is_kalshi else "polymarket_event"
    for event in events:
        changed_markets = []
        fingerprints: dict[str, str] = {}
        for market in event["markets"]:
            fingerprint = entity_input_fingerprint(
                event_title=event["title"],
                event_slug=event["slug"],
                market=market,
            )
            fingerprints[market["market_id"]] = fingerprint
            if (
                market["prior_entity_input_sha256"] == fingerprint
                and market["prior_extractor_version"] == EXTRACTOR_VERSION
            ):
                batch.skipped_unchanged_markets += 1
            else:
                changed_markets.append(market)
        for market_chunk in _chunks(changed_markets, MARKET_CHUNK_SIZE):
            try:
                provider_results = _analyze_market_chunk(
                    provider,
                    event=event,
                    markets=market_chunk,
                )
            except Exception as exc:
                batch.failures.append(
                    {
                        "source": event_label,
                        "source_id": event["event_id"],
                        "market_ids": [market["market_id"] for market in market_chunk],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            market_by_id = {market["market_id"]: market for market in market_chunk}
            for provider_result in provider_results:
                batch.input_tokens += provider_result.usage.input_tokens
                batch.output_tokens += provider_result.usage.output_tokens
                for disposition in provider_result.output.markets:
                    market = market_by_id[disposition.market_id]
                    fingerprint = fingerprints[disposition.market_id]
                    batch.classifications[disposition.market_id] = {
                        (
                            "market_ticker" if is_kalshi else "market_id"
                        ): disposition.market_id,
                        "source_content_sha256": market["source_content_sha256"],
                        "entity_input_sha256": fingerprint,
                        "market_topic": disposition.market_topic.value,
                        "contract_type": disposition.contract_type.value,
                        "extractor_version": EXTRACTOR_VERSION,
                        "confidence": disposition.confidence,
                        "classification_metadata": {
                            "provider": provider_result.provider,
                            "model_name": provider_result.model_name,
                        },
                        "updated_at": observed_at,
                    }
                    source = SourceReference(
                        source_kind=source_kind,
                        source_id=market["market_id"],
                        source_content_sha256=market["source_content_sha256"],
                        market_id=None if is_kalshi else market["market_id"],
                        kalshi_market_ticker=(
                            market["market_id"] if is_kalshi else None
                        ),
                    )
                    source_fields = {
                        "event_title": str(event["title"] or ""),
                        "event_slug": str(event.get("slug") or ""),
                        "question": str(market["question"] or ""),
                        "market_slug": str(market.get("slug") or ""),
                        "group_item_title": str(
                            market.get("group_item_title") or ""
                        ),
                        "group_item_threshold": str(
                            market.get("group_item_threshold") or ""
                        ),
                        "outcomes": " | ".join(market.get("outcomes") or []),
                        "sports_market_type": str(
                            market.get("sports_market_type") or ""
                        ),
                    }
                    context = " | ".join(
                        value for value in source_fields.values() if value
                    )
                    processed_mentions: set[
                        tuple[str, EntityType, MentionRole]
                    ] = set()
                    group_item = market.get("group_item_title")
                    if group_item:
                        ignored_reason = placeholder_reason(group_item)
                        if ignored_reason:
                            ignored = ExtractedMention(
                                text=group_item,
                                entity_type=EntityType.PERSON,
                                person_role_hint=PersonRoleHint.UNKNOWN,
                                mention_role=MentionRole.CANDIDATE,
                                evidence=group_item,
                                confidence=1,
                                source_refs=["group_item_title"],
                            )
                            row = ignored_mention(
                                text=group_item,
                                reason=ignored_reason,
                                mention=ignored,
                                source=source,
                                bank_version_id=bank_version_id,
                                observed_at=observed_at,
                            )
                            batch.mentions[row["mention_id"]] = row
                            processed_mentions.add(
                                (
                                    normalize_name(ignored.text),
                                    ignored.entity_type,
                                    ignored.mention_role,
                                )
                            )
                        elif disposition.ignore_group_item:
                            # The extractor judged the label a non-entity (e.g.
                            # Kalshi ladder phrases like "Over 10.5 1Q points
                            # scored"); record it as ignored, not a failure.
                            ignored = ExtractedMention(
                                text=group_item,
                                entity_type=EntityType.PERSON,
                                person_role_hint=PersonRoleHint.UNKNOWN,
                                mention_role=MentionRole.CANDIDATE,
                                evidence=group_item,
                                confidence=disposition.confidence,
                                source_refs=["group_item_title"],
                            )
                            row = ignored_mention(
                                text=group_item,
                                reason=(
                                    disposition.ignore_reason
                                    or "extractor_ignored_group_item"
                                ),
                                mention=ignored,
                                source=source,
                                bank_version_id=bank_version_id,
                                observed_at=observed_at,
                            )
                            batch.mentions[row["mention_id"]] = row
                            processed_mentions.add(
                                (
                                    normalize_name(ignored.text),
                                    ignored.entity_type,
                                    ignored.mention_role,
                                )
                            )
                        elif disposition.group_item_entity_type is None:
                            batch.failures.append(
                                {
                                    "source": source_kind,
                                    "source_id": market["market_id"],
                                    "error": "group item was not typed by extractor",
                                }
                            )
                        else:
                            exact_bank_matches = index.exact_matches(group_item)
                            group_entity_type = disposition.group_item_entity_type
                            group_person_role = (
                                disposition.group_item_person_role_hint
                            )
                            if len(exact_bank_matches) == 1:
                                group_entity_type = EntityType(
                                    exact_bank_matches[0]["entity_type"]
                                )
                                if group_entity_type == EntityType.TEAM:
                                    group_person_role = None
                            group_mention = ExtractedMention(
                                text=group_item,
                                entity_type=group_entity_type,
                                person_role_hint=group_person_role,
                                mention_role=(
                                    disposition.group_item_mention_role
                                    or MentionRole.CANDIDATE
                                ),
                                evidence=group_item,
                                confidence=disposition.confidence,
                                source_refs=["group_item_title"],
                            )
                            batch.add_resolution(
                                resolve_mention(
                                    mention=group_mention,
                                    source=source,
                                    source_context=context,
                                    index=index,
                                    provider=provider,
                                    bank_version_id=bank_version_id,
                                    observed_at=observed_at,
                                    allow_provisional=(
                                        group_mention.entity_type == EntityType.PERSON
                                        and group_mention.person_role_hint
                                        in PROVISIONAL_PERSON_ROLES
                                        and disposition.market_topic
                                        in PROVISIONAL_PERSON_TOPICS
                                    ),
                                )
                            )
                            processed_mentions.add(
                                (
                                    normalize_name(group_mention.text),
                                    group_mention.entity_type,
                                    group_mention.mention_role,
                                )
                            )
                    for mention in disposition.standalone_mentions:
                        try:
                            mention_key = (
                                normalize_name(mention.text),
                                mention.entity_type,
                                mention.mention_role,
                            )
                            if mention_key in processed_mentions:
                                continue
                            ignored_reason = placeholder_reason(mention.text)
                            if ignored_reason:
                                row = ignored_mention(
                                    text=mention.text,
                                    reason=ignored_reason,
                                    mention=mention,
                                    source=source,
                                    bank_version_id=bank_version_id,
                                    observed_at=observed_at,
                                )
                                batch.mentions[row["mention_id"]] = row
                                processed_mentions.add(mention_key)
                                continue
                            _validate_extracted_evidence(
                                mention,
                                source_context=context,
                                source_fields=source_fields,
                            )
                            batch.add_resolution(
                                resolve_mention(
                                    mention=mention,
                                    source=source,
                                    source_context=context,
                                    index=index,
                                    provider=provider,
                                    bank_version_id=bank_version_id,
                                    observed_at=observed_at,
                                    allow_provisional=False,
                                )
                            )
                            processed_mentions.add(mention_key)
                        except Exception as exc:
                            batch.failures.append(
                                {
                                    "source": source_kind,
                                    "source_id": market["market_id"],
                                    "mention": mention.model_dump(mode="json"),
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                    if (
                        disposition.market_topic == MarketTopic.GAME
                        and disposition.contract_type
                        in {ContractType.MONEYLINE, ContractType.SPREAD}
                    ):
                        for outcome in market.get("outcomes", []):
                            if placeholder_reason(outcome):
                                continue
                            exact_matches = index.exact_matches(outcome)
                            if (
                                len(exact_matches) != 1
                                or exact_matches[0]["entity_type"] != EntityType.TEAM
                            ):
                                continue
                            aliases = {
                                exact_matches[0]["canonical_name"],
                                *exact_matches[0].get("aliases", []),
                            }
                            if not deterministic_alias_match_is_safe(
                                outcome,
                                entity_type=EntityType.TEAM.value,
                                aliases=aliases,
                            ):
                                continue
                            outcome_mention = ExtractedMention(
                                text=outcome,
                                entity_type=EntityType.TEAM,
                                mention_role=MentionRole.COMPETITOR,
                                evidence=outcome,
                                confidence=1,
                                source_refs=["outcomes"],
                            )
                            mention_key = (
                                normalize_name(outcome),
                                EntityType.TEAM,
                                MentionRole.COMPETITOR,
                            )
                            if mention_key in processed_mentions:
                                continue
                            batch.add_resolution(
                                resolve_mention(
                                    mention=outcome_mention,
                                    source=source,
                                    source_context=context,
                                    index=index,
                                    provider=provider,
                                    bank_version_id=bank_version_id,
                                    observed_at=observed_at,
                                    allow_provisional=False,
                                )
                            )
                            processed_mentions.add(mention_key)


def process_news(
    *,
    records: list[dict[str, Any]],
    provider: EntityProvider,
    index: CandidateIndex,
    bank_version_id: str | None,
    batch: Batch,
    observed_at: datetime,
) -> None:
    for record in records:
        failure_count_before = len(batch.failures)
        mention_count_before = len(batch.mentions)
        try:
            source = SourceReference(
                source_kind="news",
                source_id=record["news_id"],
                source_content_sha256=record["input_fingerprint"],
                news_id=record["news_id"],
            )
            for raw_entity in record["entities"] or []:
                source_type = EnrichmentEntityType(raw_entity["entity_type"])
                if source_type == EnrichmentEntityType.TEAM:
                    entity_type = EntityType.TEAM
                    person_role_hint = None
                elif source_type == EnrichmentEntityType.PLAYER:
                    entity_type = EntityType.PERSON
                    person_role_hint = PersonRoleHint.PLAYER
                elif source_type == EnrichmentEntityType.COACH:
                    entity_type = EntityType.PERSON
                    person_role_hint = PersonRoleHint.COACH
                else:
                    continue
                mention = ExtractedMention(
                    text=raw_entity["name"],
                    entity_type=entity_type,
                    person_role_hint=person_role_hint,
                    mention_role=raw_entity["mention_role"],
                    evidence=raw_entity["evidence"],
                    confidence=float(raw_entity["confidence"]),
                    source_refs=list(raw_entity["source_refs"]),
                )
                batch.add_resolution(
                    resolve_mention(
                        mention=mention,
                        source=source,
                        source_context=" | ".join(
                            value
                            for value in (
                                record["text"],
                                record.get("summary"),
                                mention.evidence,
                            )
                            if value
                        ),
                        index=index,
                        provider=provider,
                        bank_version_id=bank_version_id,
                        observed_at=observed_at,
                        allow_provisional=False,
                        extractor_version=ENTITY_EXTRACTOR_VERSION,
                    )
                )
        except Exception as exc:
            batch.failures.append(
                {
                    "source": "news",
                    "source_id": record["news_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        failure_count = len(batch.failures) - failure_count_before
        mention_count = len(batch.mentions) - mention_count_before
        run_key = (
            record["news_id"],
            record["enrichment_version"],
            record["input_fingerprint"],
            ENTITY_EXTRACTOR_VERSION,
        )
        batch.news_resolution_runs[run_key] = {
            "news_id": record["news_id"],
            "enrichment_version": record["enrichment_version"],
            "input_fingerprint": record["input_fingerprint"],
            "extractor_version": ENTITY_EXTRACTOR_VERSION,
            "bank_version_id": bank_version_id,
            "status": "failed" if failure_count else "completed",
            "mention_count": mention_count,
            "failure_count": failure_count,
            "started_at": observed_at,
            "completed_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }


def process_pending_mentions(
    *,
    rows: list[dict[str, Any]],
    provider: EntityProvider,
    index: CandidateIndex,
    bank_version_id: str,
    batch: Batch,
    observed_at: datetime,
) -> None:
    for row in rows:
        mention = ExtractedMention(
            text=row["mention_text"],
            entity_type=row["entity_type_hint"],
            person_role_hint=row["person_role_hint"],
            mention_role=row["mention_role"],
            evidence=row["evidence"],
            confidence=float(row["confidence"]),
            source_refs=list(row.get("source_refs") or []),
        )
        current_candidates = index.retrieve(mention)
        current_candidate_ids = sorted(
            candidate.entity_id for candidate in current_candidates
        )
        if current_candidate_ids == sorted(row["candidate_entity_ids"] or []):
            refreshed = dict(row)
            refreshed["last_bank_version_id"] = bank_version_id
            refreshed["updated_at"] = observed_at
            batch.mentions[refreshed["mention_id"]] = refreshed
            batch.skipped_unchanged_candidate_sets += 1
            continue
        if row["news_id"]:
            source_kind = "news"
            source_id = row["news_id"]
        elif row["polymarket_market_id"]:
            source_kind = "polymarket_market"
            source_id = row["polymarket_market_id"]
        elif row.get("kalshi_market_ticker"):
            source_kind = "kalshi_market"
            source_id = row["kalshi_market_ticker"]
        else:
            source_kind = "polymarket_event"
            source_id = row["polymarket_event_id"]
        source = SourceReference(
            source_kind=source_kind,
            source_id=source_id,
            source_content_sha256=row["source_content_sha256"],
            news_id=row["news_id"],
            event_id=row["polymarket_event_id"],
            market_id=row["polymarket_market_id"],
            kalshi_market_ticker=row.get("kalshi_market_ticker"),
        )
        batch.add_resolution(
            resolve_mention(
                mention=mention,
                source=source,
                source_context=row["evidence"],
                index=index,
                provider=provider,
                bank_version_id=bank_version_id,
                observed_at=observed_at,
                allow_provisional=False,
                extractor_version=row["extractor_version"],
            )
        )


def _write_records(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(serialize_audit_record(record))
            output.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.event_limit <= 100:
        print("ERROR: --event-limit must be between 1 and 100", file=sys.stderr)
        return 2
    if not 1 <= args.news_limit <= 100:
        print("ERROR: --news-limit must be between 1 and 100", file=sys.stderr)
        return 2
    if not 1 <= args.retry_limit <= 1000:
        print("ERROR: --retry-limit must be between 1 and 1000", file=sys.stderr)
        return 2
    if args.apply and args.provider != "claude":
        print("ERROR: --apply requires --provider claude", file=sys.stderr)
        return 2
    if args.apply and args.confirm_live_writes != WRITE_CONFIRMATION:
        print(
            f"ERROR: --apply requires --confirm-live-writes {WRITE_CONFIRMATION}",
            file=sys.stderr,
        )
        return 2

    project_root = Path(__file__).resolve().parents[2]
    src_dir = project_root / "src"
    settings = load_enrichment_settings(src_dir)
    if args.provider == "claude":
        if not settings.api_key:
            print("ERROR: ANTHROPIC_API_KEY is not configured", file=sys.stderr)
            return 2
        provider: EntityProvider = ClaudeEntityProvider(
            settings.api_key,
            model_name=args.model or settings.model_name,
            max_tokens=settings.max_output_tokens,
        )
    else:
        provider = DeterministicEntityProvider()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else project_root / "data" / "local" / "entity_bank" / f"resolve_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    repository = ResolutionRepository.from_environment(src_dir)
    observed_at = datetime.now(UTC)
    try:
        bank_version_id = repository.latest_bank_version_id()
        if args.apply and not bank_version_id:
            print("ERROR: apply a canonical entity-bank source before resolution", file=sys.stderr)
            return 2
        index = CandidateIndex(repository.load_candidate_rows())
        batch = Batch()
        loaded_events = 0
        loaded_news = 0
        retried_mentions = 0
        if bank_version_id:
            pending_mentions = repository.load_pending_mentions(
                bank_version_id=bank_version_id,
                limit=args.retry_limit,
            )
            retried_mentions = len(pending_mentions)
            process_pending_mentions(
                rows=pending_mentions,
                provider=provider,
                index=index,
                bank_version_id=bank_version_id,
                batch=batch,
                observed_at=observed_at,
            )
        if args.source in {"markets", "both"}:
            events = repository.load_market_events(event_limit=args.event_limit)
            loaded_events = len(events)
            process_market_events(
                events=events,
                provider=provider,
                index=index,
                bank_version_id=bank_version_id,
                batch=batch,
                observed_at=observed_at,
            )
        if args.source in {"news", "both"}:
            records = repository.load_news(
                limit=args.news_limit,
                extractor_version=ENTITY_EXTRACTOR_VERSION,
                enrichment_version=settings.enrichment_version,
            )
            loaded_news = len(records)
            process_news(
                records=records,
                provider=provider,
                index=index,
                bank_version_id=bank_version_id,
                batch=batch,
                observed_at=observed_at,
            )
        if args.apply:
            repository.persist_batch(batch.as_repository_batch())
    finally:
        repository.close()

    _write_records(output_dir / "classifications.jsonl", batch.classifications.values())
    _write_records(output_dir / "mentions.jsonl", batch.mentions.values())
    _write_records(
        output_dir / "provisional_entities.jsonl",
        batch.provisional_entities.values(),
    )
    _write_records(output_dir / "resolution_attempts.jsonl", batch.attempts.values())
    _write_records(
        output_dir / "news_resolution_runs.jsonl",
        batch.news_resolution_runs.values(),
    )
    _write_records(output_dir / "failures.jsonl", batch.failures)
    summary = {
        "dry_run": not args.apply,
        "database_reads": True,
        "database_writes": args.apply,
        "gcs_reads": False,
        "gcs_writes": False,
        "source": args.source,
        "provider": provider.provider_name,
        "model_name": provider.model_name,
        "extractor_version": EXTRACTOR_VERSION,
        "resolver_version": RESOLVER_VERSION,
        "bank_version_id": bank_version_id,
        "loaded_events": loaded_events,
        "loaded_news": loaded_news,
        "news_resolution_runs": len(batch.news_resolution_runs),
        "pending_mentions_considered": retried_mentions,
        "classifications": len(batch.classifications),
        "mentions": len(batch.mentions),
        "resolved": sum(
            row["resolution_status"] == "resolved" for row in batch.mentions.values()
        ),
        "ambiguous": sum(
            row["resolution_status"] == "ambiguous" for row in batch.mentions.values()
        ),
        "unresolved": sum(
            row["resolution_status"] == "unresolved" for row in batch.mentions.values()
        ),
        "ignored": sum(
            row["resolution_status"] == "ignored" for row in batch.mentions.values()
        ),
        "provisional_entities": len(batch.provisional_entities),
        "skipped_unchanged_markets": batch.skipped_unchanged_markets,
        "skipped_unchanged_candidate_sets": (
            batch.skipped_unchanged_candidate_sets
        ),
        "failures": len(batch.failures),
        "input_tokens": batch.input_tokens,
        "output_tokens": batch.output_tokens,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if batch.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
