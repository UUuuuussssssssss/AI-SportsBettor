"""Anthropic and deterministic dry-run enrichment providers."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Protocol

from anthropic import Anthropic
from pydantic import ValidationError

from src.enrich_news.config import DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MODEL_NAME
from src.enrich_news.models import (
    BatchedEnrichmentOutput,
    EnrichmentOutput,
    ExtractedClaim,
    InformationStatus,
    ProviderUsage,
    TagAssignment,
    TagCertainty,
    TopicTag,
    Usefulness,
)
from src.enrich_news.prompt import SYSTEM_PROMPT, build_batch_user_prompt, build_user_prompt
from src.enrich_news.sources import CollectedEvidence, ImagePayload

BATCH_TIMEOUT_SECONDS = 180.0
BATCH_OUTPUT_TOKENS_PER_ITEM = 2_048
MAX_BATCH_OUTPUT_TOKENS = 16_384
BATCH_SYSTEM_ADDENDUM = (
    "\nYou are scoring several independent tweets in one response. "
    "Keep each tweet's extraction isolated."
)


@dataclass(frozen=True)
class ProviderResponse:
    output: EnrichmentOutput
    usage: ProviderUsage
    model_name: str


class EnrichmentProvider(Protocol):
    provider_name: str
    model_name: str

    def enrich(self, evidence: CollectedEvidence) -> ProviderResponse: ...


def split_usage(usage: ProviderUsage, count: int) -> list[ProviderUsage]:
    """Divide one API call's tokens across the tweets scored in that call."""

    if count < 1:
        raise ValueError("count must be at least 1")
    shares: list[ProviderUsage] = []
    for index in range(count):
        shares.append(
            ProviderUsage(
                input_tokens=_share_tokens(usage.input_tokens, count, index),
                output_tokens=_share_tokens(usage.output_tokens, count, index),
                cache_creation_input_tokens=_share_tokens(
                    usage.cache_creation_input_tokens, count, index
                ),
                cache_read_input_tokens=_share_tokens(
                    usage.cache_read_input_tokens, count, index
                ),
            )
        )
    return shares


def _share_tokens(total: int, count: int, index: int) -> int:
    base, remainder = divmod(total, count)
    return base + (remainder if index == 0 else 0)


def _namespaced_prompt_parts(
    news_id: str,
    evidence: CollectedEvidence,
) -> tuple[str, list[str], list[ImagePayload]]:
    mapping = {ref: f"{news_id}/{ref}" for ref in evidence.source_refs()}
    text = evidence.as_prompt_text()
    for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
        text = text.replace(f"[{old}]", f"[{new}]")
    images = [
        ImagePayload(
            source_ref=mapping.get(image.source_ref, f"{news_id}/{image.source_ref}"),
            media_type=image.media_type,
            data=image.data,
            sha256=image.sha256,
        )
        for image in evidence.images
    ]
    return text, [mapping[ref] for ref in evidence.source_refs()], images


def _denamespace_output(news_id: str, output: EnrichmentOutput) -> EnrichmentOutput:
    prefix = f"{news_id}/"
    for item in [*output.tags, *output.entities, *output.claims]:
        item.source_refs = [ref.removeprefix(prefix) for ref in item.source_refs]
    return output


class ClaudeProvider:
    provider_name = "anthropic"

    def __init__(
        self,
        api_key: str,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        timeout_seconds: float = 90.0,
    ) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the Claude provider")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = Anthropic(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=2,
        )

    def enrich(self, evidence: CollectedEvidence) -> ProviderResponse:
        return self._enrich(evidence)

    def repair(
        self,
        evidence: CollectedEvidence,
        validation_feedback: str,
    ) -> ProviderResponse:
        correction = (
            "\n\nThe previous extraction had invalid entity evidence. Regenerate the complete "
            "answer, copying every entity evidence excerpt exactly from its cited source. Do not "
            "combine separate passages or decode, paraphrase, summarize, or normalize the excerpt. "
            f"Validation feedback: {validation_feedback[:4_000]}"
        )
        return self._enrich(evidence, system_addendum=correction)

    def enrich_many(
        self,
        items: list[tuple[str, CollectedEvidence]],
    ) -> dict[str, ProviderResponse]:
        if not items:
            return {}
        if len(items) == 1:
            news_id, evidence = items[0]
            return {news_id: self.enrich(evidence)}

        prompt_items: list[tuple[str, str, list[str]]] = []
        images: list[ImagePayload] = []
        for news_id, evidence in items:
            text, source_refs, tweet_images = _namespaced_prompt_parts(news_id, evidence)
            prompt_items.append((news_id, text, source_refs))
            images.extend(tweet_images[:20])

        content: list[dict[str, object]] = []
        for image in images:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image.media_type,
                        "data": base64.b64encode(image.data).decode("ascii"),
                    },
                }
            )
            content.append(
                {
                    "type": "text",
                    "text": f"The preceding image has source reference [{image.source_ref}].",
                }
            )
        content.append({"type": "text", "text": build_batch_user_prompt(prompt_items)})
        max_tokens = min(
            MAX_BATCH_OUTPUT_TOKENS,
            max(self.max_tokens, BATCH_OUTPUT_TOKENS_PER_ITEM * len(items)),
        )
        request = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
            "output_format": BatchedEnrichmentOutput,
        }
        system_prompt = SYSTEM_PROMPT + BATCH_SYSTEM_ADDENDUM
        client = self.client
        with_options = getattr(client, "with_options", None)
        if callable(with_options):
            client = with_options(timeout=BATCH_TIMEOUT_SECONDS)
        try:
            response = client.messages.parse(system=system_prompt, **request)
        except ValidationError as exc:
            correction = (
                "\n\nYour previous response failed output-schema validation. Regenerate the "
                "complete batched answer from the supplied evidence and satisfy every schema "
                f"limit. Validation error: {str(exc)[:4_000]}"
            )
            response = client.messages.parse(
                system=system_prompt + correction,
                **request,
            )
        if response.parsed_output is None:
            raise RuntimeError(f"Claude returned no structured output ({response.stop_reason})")
        parsed: BatchedEnrichmentOutput = response.parsed_output
        expected = {news_id for news_id, _evidence in items}
        usage = ProviderUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_creation_input_tokens=(response.usage.cache_creation_input_tokens or 0),
            cache_read_input_tokens=response.usage.cache_read_input_tokens or 0,
        )
        ordered: list[tuple[str, EnrichmentOutput]] = []
        seen: set[str] = set()
        for item in parsed.tweets:
            if item.news_id not in expected or item.news_id in seen:
                continue
            seen.add(item.news_id)
            ordered.append((item.news_id, _denamespace_output(item.news_id, item.to_output())))
        shares = split_usage(usage, len(ordered) or 1)
        return {
            news_id: ProviderResponse(
                output=output,
                usage=shares[index],
                model_name=self.model_name,
            )
            for index, (news_id, output) in enumerate(ordered)
        }

    def _enrich(
        self,
        evidence: CollectedEvidence,
        *,
        system_addendum: str = "",
    ) -> ProviderResponse:
        content: list[dict[str, object]] = []
        for image in evidence.images[:20]:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image.media_type,
                        "data": base64.b64encode(image.data).decode("ascii"),
                    },
                }
            )
            content.append(
                {
                    "type": "text",
                    "text": f"The preceding image has source reference [{image.source_ref}].",
                }
            )
        content.append(
            {
                "type": "text",
                "text": build_user_prompt(evidence.as_prompt_text(), evidence.source_refs()),
            }
        )
        request = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content}],
            "output_format": EnrichmentOutput,
        }
        system_prompt = SYSTEM_PROMPT + system_addendum
        try:
            response = self.client.messages.parse(system=system_prompt, **request)
        except ValidationError as exc:
            correction = (
                "\n\nYour previous response failed output-schema validation. Regenerate the "
                "complete answer from the supplied evidence and satisfy every schema limit. "
                f"Validation error: {str(exc)[:4_000]}"
            )
            response = self.client.messages.parse(
                system=system_prompt + correction,
                **request,
            )
        if response.parsed_output is None:
            raise RuntimeError(f"Claude returned no structured output ({response.stop_reason})")
        return ProviderResponse(
            output=response.parsed_output,
            usage=ProviderUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_creation_input_tokens=(response.usage.cache_creation_input_tokens or 0),
                cache_read_input_tokens=response.usage.cache_read_input_tokens or 0,
            ),
            model_name=self.model_name,
        )


_TAG_KEYWORDS: tuple[tuple[TopicTag, tuple[str, ...]], ...] = (
    (
        TopicTag.INJURY_AVAILABILITY,
        (
            "injury",
            "injured",
            "questionable",
            "doubtful",
            "ruled out",
            "did not practice",
            "limited practice",
            "left practice",
            "game-time decision",
        ),
    ),
    (
        TopicTag.LINEUP_DEPTH_CHART,
        ("depth chart", "starter", "starting", "benched", "backup", "snap count", "role"),
    ),
    (
        TopicTag.ROSTER_TRANSACTION,
        ("trade", "traded", "acquired", "signed", "released", "waived", "activated"),
    ),
    (
        TopicTag.COACHING_MANAGEMENT,
        ("head coach", "offensive coordinator", "defensive coordinator", "fired", "hired"),
    ),
    (
        TopicTag.CONTRACT,
        ("contract", "extension", "holdout", "salary", "guaranteed"),
    ),
    (
        TopicTag.DISCIPLINE_LEGAL,
        ("suspended", "suspension", "arrested", "investigation", "fine", "lawsuit"),
    ),
    (
        TopicTag.WEATHER_FIELD_CONDITIONS,
        ("weather", "rain", "snow", "wind", "roof", "field conditions", "turf"),
    ),
    (
        TopicTag.SCHEDULE_TRAVEL,
        ("rescheduled", "schedule", "travel", "flight", "short week", "bye week"),
    ),
    (
        TopicTag.GAME_STATUS_RESULT,
        ("final score", "final:", "postponed", "cancelled", "halftime", "overtime"),
    ),
    (
        TopicTag.PERFORMANCE_STATISTICS,
        ("yards", "touchdowns", "interceptions", "record", "completion percentage"),
    ),
    (
        TopicTag.MARKET_ODDS,
        ("betting odds", "betting line", "point spread", "moneyline", "line movement"),
    ),
    (
        TopicTag.LEAGUE_MANAGEMENT_RULE,
        ("rule change", "league policy", "officiating", "competition committee"),
    ),
)
_PROMOTIONAL_KEYWORDS = ("subscribe", "giveaway", "merch", "tickets on sale", "happy birthday")


def _contains_keyword(text: str, keyword: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text))


class DeterministicDryRunProvider:
    """Offline control-flow test double; its output is never presented as model output."""

    provider_name = "deterministic_dry_run"
    model_name = "keyword-fixture-v1"

    def enrich(self, evidence: CollectedEvidence) -> ProviderResponse:
        text = evidence.as_prompt_text()
        lowered = text.lower()
        matched = [
            topic
            for topic, keywords in _TAG_KEYWORDS
            if any(_contains_keyword(lowered, keyword) for keyword in keywords)
        ]
        if not matched and any(
            _contains_keyword(lowered, keyword) for keyword in _PROMOTIONAL_KEYWORDS
        ):
            matched = [TopicTag.PROMOTIONAL_SOCIAL]
        if not matched:
            matched = [TopicTag.UNRELATED_OTHER]

        high_usefulness_tags = {
            TopicTag.INJURY_AVAILABILITY,
            TopicTag.LINEUP_DEPTH_CHART,
            TopicTag.ROSTER_TRANSACTION,
            TopicTag.COACHING_MANAGEMENT,
            TopicTag.DISCIPLINE_LEGAL,
            TopicTag.WEATHER_FIELD_CONDITIONS,
            TopicTag.SCHEDULE_TRAVEL,
        }
        medium_usefulness_tags = {
            TopicTag.CONTRACT,
            TopicTag.GAME_STATUS_RESULT,
            TopicTag.PERFORMANCE_STATISTICS,
            TopicTag.MARKET_ODDS,
            TopicTag.LEAGUE_MANAGEMENT_RULE,
        }
        if any(tag in high_usefulness_tags for tag in matched):
            usefulness = Usefulness.HIGH
        elif any(tag in medium_usefulness_tags for tag in matched):
            usefulness = Usefulness.MEDIUM
        else:
            usefulness = Usefulness.IRRELEVANT

        if any(value in lowered for value in ("rumor", "could be", "might be")):
            information_status = InformationStatus.RUMOR
        elif any(value in lowered for value in ("i think", "in my opinion", "analysis:")):
            information_status = InformationStatus.OPINION
        elif any(value in lowered for value in ("official", "team announced", "we have signed")):
            information_status = InformationStatus.OFFICIAL
        else:
            information_status = InformationStatus.REPORTED

        tweet_match = re.search(r"Tweet text:\s*(.+)", text)
        summary_source = tweet_match.group(1).strip() if tweet_match else text.strip()
        summary = summary_source[:300] or "No usable source text."
        output = EnrichmentOutput(
            tags=[
                TagAssignment(
                    tag=tag,
                    certainty=TagCertainty.NEUTRAL,
                    source_refs=["tweet"],
                )
                for tag in matched
            ],
            information_status=information_status,
            usefulness=usefulness,
            summary=summary,
            classification_reason=(
                "Deterministic keyword fixture used for an offline pipeline dry run; "
                "this is not an AI quality judgment."
            ),
            entities=[],
            claims=[
                ExtractedClaim(
                    statement=summary,
                    confidence=0.65,
                    source_refs=["tweet"],
                )
            ],
        )
        return ProviderResponse(
            output=output,
            usage=ProviderUsage(),
            model_name=self.model_name,
        )

    def enrich_many(
        self,
        items: list[tuple[str, CollectedEvidence]],
    ) -> dict[str, ProviderResponse]:
        return {news_id: self.enrich(evidence) for news_id, evidence in items}
