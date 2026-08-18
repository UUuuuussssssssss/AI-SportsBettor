"""Orchestration for one independently auditable enrichment."""

from __future__ import annotations

import hashlib
import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime

from src.enrich_news.models import (
    EnrichmentOutput,
    EnrichmentResult,
    ExtractedEntity,
    NewsRecord,
    ProviderUsage,
)
from src.enrich_news.prompt import ENTITY_EXTRACTOR_VERSION, PROMPT_VERSION
from src.enrich_news.provider import EnrichmentProvider, ProviderResponse
from src.enrich_news.sources import CollectedEvidence, collect_evidence
from src.entity_bank.normalization import normalize_name


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _normalize_source_refs(
    output: EnrichmentOutput,
    aliases: dict[str, str],
) -> None:
    """Replace exact known aliases while preserving order and rejecting all others later."""

    for item in [*output.tags, *output.entities, *output.claims]:
        normalized = (
            aliases.get(source_ref.strip(), source_ref.strip()) for source_ref in item.source_refs
        )
        item.source_refs = list(dict.fromkeys(normalized))


def _supporting_excerpt(
    evidence: CollectedEvidence,
    *,
    entity_name: str,
    source_refs: list[str],
) -> str | None:
    normalized_name = normalize_name(html.unescape(entity_name))
    allowed_refs = set(source_refs)
    for section in evidence.text_sections:
        first_line = section.splitlines()[0] if section else ""
        source_ref = (
            first_line[1 : first_line.index("]")]
            if first_line.startswith("[") and "]" in first_line
            else None
        )
        if source_ref not in allowed_refs:
            continue
        for line in section.splitlines():
            candidate = line.split(": ", 1)[-1].strip()
            if normalized_name not in normalize_name(html.unescape(candidate)):
                continue
            if len(candidate) <= 2_000:
                return candidate
            raw_index = candidate.casefold().find(entity_name.casefold())
            if raw_index < 0:
                return candidate[:2_000]
            start = max(0, raw_index - 900)
            end = min(len(candidate), raw_index + len(entity_name) + 900)
            return candidate[start:end]
    return None


def _validate_source_refs(
    output: EnrichmentOutput,
    evidence: CollectedEvidence,
    aliases: dict[str, str],
) -> None:
    _normalize_source_refs(output, aliases)
    valid_source_refs = set(evidence.source_refs())
    used_source_refs = {
        source_ref
        for item in [*output.tags, *output.entities, *output.claims]
        for source_ref in item.source_refs
    }
    unknown_refs = sorted(used_source_refs - valid_source_refs)
    if unknown_refs:
        raise ValueError(f"provider returned unknown source references: {unknown_refs}")


def _repair_entity_evidence(
    output: EnrichmentOutput,
    evidence: CollectedEvidence,
) -> list[tuple[ExtractedEntity, str]]:
    evidence_text = evidence.as_prompt_text()
    normalized_raw_evidence = normalize_name(evidence_text)
    normalized_evidence = normalize_name(html.unescape(evidence_text))
    invalid: list[tuple[ExtractedEntity, str]] = []
    for entity in output.entities:
        normalized_name = normalize_name(html.unescape(entity.name))
        normalized_raw_excerpt = normalize_name(entity.evidence)
        normalized_excerpt = normalize_name(html.unescape(entity.evidence))
        if normalized_name not in normalized_evidence:
            invalid.append((entity, "entity name is absent from collected evidence"))
            continue
        excerpt_is_supported = normalized_raw_excerpt in normalized_raw_evidence
        excerpt_is_html_equivalent = normalized_excerpt in normalized_evidence
        excerpt_identifies_entity = normalized_name in normalized_excerpt
        if excerpt_is_supported and excerpt_identifies_entity:
            continue
        repaired = _supporting_excerpt(
            evidence,
            entity_name=entity.name,
            source_refs=entity.source_refs,
        )
        if repaired is None:
            reason = (
                "no raw excerpt found for HTML-equivalent evidence"
                if excerpt_is_html_equivalent
                else "no verbatim excerpt found in the cited source"
            )
            invalid.append((entity, reason))
            continue
        entity.evidence = repaired
        evidence.warnings.append(f"entity evidence repaired from cited source: {entity.name}")
    return invalid


def _usage_sum(first: ProviderUsage, second: ProviderUsage) -> ProviderUsage:
    return ProviderUsage(
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        cache_creation_input_tokens=(
            first.cache_creation_input_tokens + second.cache_creation_input_tokens
        ),
        cache_read_input_tokens=(first.cache_read_input_tokens + second.cache_read_input_tokens),
    )


@dataclass
class _PreparedEnrichment:
    record: NewsRecord
    evidence: CollectedEvidence
    input_manifest: dict
    input_fingerprint: str
    started_at: datetime


def _prepare_record(record: NewsRecord, *, allow_network: bool) -> _PreparedEnrichment:
    started_at = datetime.now(UTC)
    evidence = collect_evidence(record, allow_network=allow_network)
    if len(evidence.images) > 20:
        evidence.images = evidence.images[:20]
        evidence.warnings.append("image input limit reached; only the first 20 were analyzed")
    input_manifest = {
        "prompt_version": PROMPT_VERSION,
        "network_enabled": allow_network,
        **evidence.manifest,
        "image_inputs": [
            {
                "source_ref": image.source_ref,
                "media_type": image.media_type,
                "content_sha256": image.sha256,
                "byte_size": len(image.data),
            }
            for image in evidence.images
        ],
    }
    input_fingerprint = hashlib.sha256(
        _canonical_bytes(
            {
                "record": record.model_dump(mode="json"),
                "input_manifest": input_manifest,
                "evidence_text": evidence.as_prompt_text(),
            }
        )
    ).hexdigest()
    return _PreparedEnrichment(
        record=record,
        evidence=evidence,
        input_manifest=input_manifest,
        input_fingerprint=input_fingerprint,
        started_at=started_at,
    )


def _failed_result(
    prepared: _PreparedEnrichment,
    provider: EnrichmentProvider,
    *,
    enrichment_version: str,
    response: ProviderResponse | None,
    usage: ProviderUsage,
    exc: Exception,
) -> EnrichmentResult:
    return EnrichmentResult(
        news_id=prepared.record.news_id,
        enrichment_version=enrichment_version,
        entity_extractor_version=ENTITY_EXTRACTOR_VERSION,
        provider=provider.provider_name,
        model_name=response.model_name if response else provider.model_name,
        status="failed",
        input_fingerprint=prepared.input_fingerprint,
        input_manifest=prepared.input_manifest,
        usage=usage,
        warnings=prepared.evidence.warnings,
        error=f"{type(exc).__name__}: {str(exc)[:4_000]}",
        started_at=prepared.started_at,
        completed_at=datetime.now(UTC),
    )


def _finalize_result(
    prepared: _PreparedEnrichment,
    provider: EnrichmentProvider,
    response: ProviderResponse,
    *,
    enrichment_version: str,
    usage: ProviderUsage,
) -> EnrichmentResult:
    evidence = prepared.evidence
    record = prepared.record
    source_aliases = {record.source_url: "tweet"} if record.source_url else {}
    _validate_source_refs(response.output, evidence, source_aliases)
    invalid = _repair_entity_evidence(response.output, evidence)
    repair = getattr(provider, "repair", None)
    if invalid and callable(repair):
        feedback = "; ".join(f"{entity.name!r}: {reason}" for entity, reason in invalid)
        try:
            retry = repair(evidence, feedback)
            usage = _usage_sum(usage, retry.usage)
            _validate_source_refs(retry.output, evidence, source_aliases)
            retry_invalid = _repair_entity_evidence(retry.output, evidence)
            retry_invalid_ids = {id(entity) for entity, _reason in retry_invalid}
            retry_entities = {
                (
                    normalize_name(html.unescape(entity.name)),
                    entity.entity_type,
                    entity.mention_role,
                ): entity
                for entity in retry.output.entities
                if id(entity) not in retry_invalid_ids
            }
            invalid_ids = {id(entity) for entity, _reason in invalid}
            unresolved: list[tuple[ExtractedEntity, str]] = []
            repaired_entities = []
            for entity in response.output.entities:
                if id(entity) not in invalid_ids:
                    repaired_entities.append(entity)
                    continue
                key = (
                    normalize_name(html.unescape(entity.name)),
                    entity.entity_type,
                    entity.mention_role,
                )
                replacement = retry_entities.get(key)
                if replacement is None:
                    unresolved.append(
                        (entity, "validation-repair retry did not return valid evidence")
                    )
                else:
                    repaired_entities.append(replacement)
                    evidence.warnings.append(
                        f"entity evidence repaired by provider retry: {entity.name}"
                    )
            response.output.entities = repaired_entities
            invalid = unresolved
        except Exception as exc:
            evidence.warnings.append(
                f"entity evidence retry failed: {type(exc).__name__}: {str(exc)[:1_000]}"
            )
    if invalid:
        invalid_ids = {id(entity) for entity, _reason in invalid}
        response.output.entities = [
            entity for entity in response.output.entities if id(entity) not in invalid_ids
        ]
        evidence.warnings.extend(
            f"entity dropped after evidence validation: {entity.name} ({reason})"
            for entity, reason in invalid
        )
    return EnrichmentResult(
        news_id=record.news_id,
        enrichment_version=enrichment_version,
        entity_extractor_version=ENTITY_EXTRACTOR_VERSION,
        provider=provider.provider_name,
        model_name=response.model_name,
        status="completed_with_warnings" if evidence.warnings else "completed",
        input_fingerprint=prepared.input_fingerprint,
        input_manifest=prepared.input_manifest,
        output=response.output,
        usage=usage,
        warnings=evidence.warnings,
        started_at=prepared.started_at,
        completed_at=datetime.now(UTC),
    )


def enrich_record(
    record: NewsRecord,
    provider: EnrichmentProvider,
    *,
    enrichment_version: str = "v1",
    allow_network: bool = False,
) -> EnrichmentResult:
    prepared = _prepare_record(record, allow_network=allow_network)
    response = None
    usage = ProviderUsage()
    try:
        response = provider.enrich(prepared.evidence)
        usage = response.usage
        return _finalize_result(
            prepared,
            provider,
            response,
            enrichment_version=enrichment_version,
            usage=usage,
        )
    except Exception as exc:
        return _failed_result(
            prepared,
            provider,
            enrichment_version=enrichment_version,
            response=response,
            usage=usage,
            exc=exc,
        )


def enrich_records(
    records: list[NewsRecord],
    provider: EnrichmentProvider,
    *,
    enrichment_version: str = "v1",
    allow_network: bool = False,
) -> list[EnrichmentResult]:
    """Enrich one or more tweets, using one provider call when the provider supports it."""

    if len(records) <= 1:
        return [
            enrich_record(
                record,
                provider,
                enrichment_version=enrichment_version,
                allow_network=allow_network,
            )
            for record in records
        ]
    enrich_many = getattr(provider, "enrich_many", None)
    if not callable(enrich_many):
        return [
            enrich_record(
                record,
                provider,
                enrichment_version=enrichment_version,
                allow_network=allow_network,
            )
            for record in records
        ]

    prepared_list = [_prepare_record(record, allow_network=allow_network) for record in records]
    try:
        by_news_id = enrich_many(
            [(prepared.record.news_id, prepared.evidence) for prepared in prepared_list]
        )
    except Exception as exc:
        return [
            _failed_result(
                prepared,
                provider,
                enrichment_version=enrichment_version,
                response=None,
                usage=ProviderUsage(),
                exc=exc,
            )
            for prepared in prepared_list
        ]

    results: list[EnrichmentResult] = []
    for prepared in prepared_list:
        response = by_news_id.get(prepared.record.news_id)
        if response is None:
            results.append(
                _failed_result(
                    prepared,
                    provider,
                    enrichment_version=enrichment_version,
                    response=None,
                    usage=ProviderUsage(),
                    exc=ValueError(
                        f"batch response missing news_id: {prepared.record.news_id}"
                    ),
                )
            )
            continue
        try:
            results.append(
                _finalize_result(
                    prepared,
                    provider,
                    response,
                    enrichment_version=enrichment_version,
                    usage=response.usage,
                )
            )
        except Exception as exc:
            results.append(
                _failed_result(
                    prepared,
                    provider,
                    enrichment_version=enrichment_version,
                    response=response,
                    usage=response.usage,
                    exc=exc,
                )
            )
    return results
