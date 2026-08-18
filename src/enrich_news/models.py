"""Typed inputs and structured Claude outputs for news enrichment."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.entity_bank.models import MentionRole
from src.entity_bank.normalization import normalize_name


class TopicTag(StrEnum):
    INJURY_AVAILABILITY = "injury_availability"
    LINEUP_DEPTH_CHART = "lineup_depth_chart"
    ROSTER_TRANSACTION = "roster_transaction"
    COACHING_MANAGEMENT = "coaching_management"
    CONTRACT = "contract"
    DISCIPLINE_LEGAL = "discipline_legal"
    WEATHER_FIELD_CONDITIONS = "weather_field_conditions"
    SCHEDULE_TRAVEL = "schedule_travel"
    GAME_STATUS_RESULT = "game_status_result"
    PERFORMANCE_STATISTICS = "performance_statistics"
    MARKET_ODDS = "market_odds"
    LEAGUE_MANAGEMENT_RULE = "league_management_rule"
    PROMOTIONAL_SOCIAL = "promotional_social"
    UNRELATED_OTHER = "unrelated_other"


class InformationStatus(StrEnum):
    OFFICIAL = "official"
    REPORTED = "reported"
    RUMOR = "rumor"
    OPINION = "opinion"
    UNKNOWN = "unknown"


class Usefulness(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    IRRELEVANT = "irrelevant"


class TagCertainty(StrEnum):
    CONFIDENT = "confident"
    NEUTRAL = "neutral"
    UNCONFIDENT = "unconfident"


class EntityType(StrEnum):
    PLAYER = "player"
    TEAM = "team"
    COACH = "coach"
    LEAGUE = "league"
    GAME = "game"
    LOCATION = "location"
    ORGANIZATION = "organization"
    OTHER = "other"


class MediaAttachment(BaseModel):
    model_config = ConfigDict(extra="allow")

    media_key: str
    media_type: str | None = None
    source_url: str | None = None
    preview_image_url: str | None = None
    selected_source_url: str | None = None
    gcs_uri: str | None = None
    content_type: str | None = None
    alt_text: str | None = None
    duration_ms: int | None = None
    local_path: str | None = None
    transcript: str | None = None


class PreparedArticleEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ref: str
    url: str
    title: str | None = None
    text: str


class PreparedMediaEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ref: str
    media_type: str
    description: str | None = None
    ocr_text: str | None = None
    transcript: str | None = None


class NewsRecord(BaseModel):
    """Normalized source post plus optional dry-run evidence."""

    model_config = ConfigDict(extra="allow")

    news_id: str
    text: str
    source_url: str | None = None
    author_username: str | None = None
    published_at: str | None = None
    source_entities: dict[str, Any] = Field(default_factory=dict)
    media: list[MediaAttachment] = Field(default_factory=list)
    prepared_article_evidence: list[PreparedArticleEvidence] = Field(default_factory=list)
    prepared_media_evidence: list[PreparedMediaEvidence] = Field(default_factory=list)


class TagAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tag: TopicTag
    certainty: TagCertainty
    source_refs: list[str] = Field(default_factory=list, max_length=12)


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=500)
    entity_type: EntityType
    mention_role: MentionRole
    evidence: str = Field(min_length=1, max_length=2_000)
    confidence: float = Field(ge=0, le=1)
    source_refs: list[str] = Field(min_length=1, max_length=12)


class ExtractedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1, max_length=2_000)
    confidence: float = Field(ge=0, le=1)
    source_refs: list[str] = Field(default_factory=list, max_length=12)


class EnrichmentOutput(BaseModel):
    """Provider-independent result persisted by the enrichment worker."""

    model_config = ConfigDict(extra="forbid")

    tags: list[TagAssignment] = Field(min_length=1)
    information_status: InformationStatus
    usefulness: Usefulness
    summary: str = Field(min_length=1, max_length=2_000)
    classification_reason: str = Field(min_length=1, max_length=4_000)
    entities: list[ExtractedEntity] = Field(default_factory=list, max_length=12)
    claims: list[ExtractedClaim] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_tags(self) -> EnrichmentOutput:
        tags = [assignment.tag for assignment in self.tags]
        if len(tags) != len(set(tags)):
            raise ValueError("tags must be unique")
        deduplicated: dict[tuple[str, EntityType, MentionRole], ExtractedEntity] = {}
        for entity in self.entities:
            key = (
                normalize_name(entity.name),
                entity.entity_type,
                entity.mention_role,
            )
            existing = deduplicated.get(key)
            if existing is None:
                deduplicated[key] = entity
                continue
            existing.confidence = max(existing.confidence, entity.confidence)
            existing.source_refs = list(dict.fromkeys([*existing.source_refs, *entity.source_refs]))
        self.entities = list(deduplicated.values())
        return self


class PerTweetEnrichment(BaseModel):
    """One tweet's extraction inside a batched Claude response."""

    model_config = ConfigDict(extra="forbid")

    news_id: str = Field(min_length=1)
    tags: list[TagAssignment] = Field(min_length=1)
    information_status: InformationStatus
    usefulness: Usefulness
    summary: str = Field(min_length=1, max_length=2_000)
    classification_reason: str = Field(min_length=1, max_length=4_000)
    entities: list[ExtractedEntity] = Field(default_factory=list, max_length=12)
    claims: list[ExtractedClaim] = Field(default_factory=list, max_length=8)

    def to_output(self) -> EnrichmentOutput:
        return EnrichmentOutput(
            tags=self.tags,
            information_status=self.information_status,
            usefulness=self.usefulness,
            summary=self.summary,
            classification_reason=self.classification_reason,
            entities=self.entities,
            claims=self.claims,
        )


class BatchedEnrichmentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tweets: list[PerTweetEnrichment]


class ProviderUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)


class EnrichmentResult(BaseModel):
    """Complete local/persistable result including audit metadata."""

    model_config = ConfigDict(extra="forbid")

    news_id: str
    enrichment_version: str
    entity_extractor_version: str
    provider: str
    model_name: str
    status: str
    input_fingerprint: str
    input_manifest: dict[str, Any]
    output: EnrichmentOutput | None = None
    usage: ProviderUsage = Field(default_factory=ProviderUsage)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
