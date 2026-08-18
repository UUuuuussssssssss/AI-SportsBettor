import json
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import imageio_ffmpeg
import pytest
from PIL import Image

from src.enrich_news.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL_NAME,
    load_enrichment_settings,
)
from src.enrich_news.dry_run import main as dry_run_main
from src.enrich_news.models import (
    BatchedEnrichmentOutput,
    EnrichmentOutput,
    EnrichmentResult,
    EntityType,
    ExtractedEntity,
    InformationStatus,
    NewsRecord,
    PerTweetEnrichment,
    ProviderUsage,
    TagAssignment,
    TagCertainty,
    TopicTag,
    Usefulness,
)
from src.enrich_news.pipeline import enrich_record, enrich_records
from src.enrich_news.prompt import ENTITY_EXTRACTOR_VERSION, SYSTEM_PROMPT
from src.enrich_news.provider import (
    ClaudeProvider,
    DeterministicDryRunProvider,
    ProviderResponse,
    split_usage,
)
from src.enrich_news.repository import enrichment_values, tag_values
from src.enrich_news.sources import (
    CollectedEvidence,
    DownloadedResource,
    _video_sample_timestamps,
    article_urls,
    collect_evidence,
    extract_article_text,
    validate_public_url,
)
from src.enrich_news.worker import main as worker_main
from src.entity_bank.models import MentionRole


def test_structured_output_requires_unique_unordered_tags() -> None:
    with pytest.raises(ValueError, match="tags must be unique"):
        EnrichmentOutput(
            tags=[
                TagAssignment(
                    tag=TopicTag.INJURY_AVAILABILITY,
                    certainty=TagCertainty.CONFIDENT,
                    source_refs=["tweet"],
                ),
                TagAssignment(
                    tag=TopicTag.INJURY_AVAILABILITY,
                    certainty=TagCertainty.NEUTRAL,
                    source_refs=["tweet"],
                ),
            ],
            information_status=InformationStatus.REPORTED,
            usefulness=Usefulness.HIGH,
            summary="A player left practice with an injury.",
            classification_reason="The source explicitly reports an injury and practice exit.",
        )


def test_enrichment_contract_preserves_modality_and_extracts_irrelevant_entities() -> None:
    assert "affirmative claim" in SYSTEM_PROMPT
    assert "sarcastic or joking language" in SYSTEM_PROMPT
    assert '"Player X wants a trade?!" does not establish' in SYSTEM_PROMPT
    assert "must not state the rumored proposition as fact" in SYSTEM_PROMPT
    assert "Usefulness and topic classification must never suppress" in SYSTEM_PROMPT
    schema = EnrichmentOutput.model_json_schema()
    summary_schema = schema["properties"]["summary"]
    entity_schema = schema["$defs"]["ExtractedEntity"]["properties"]
    claim_schema = schema["$defs"]["ExtractedClaim"]["properties"]
    assert summary_schema["maxLength"] == 2_000
    assert schema["properties"]["classification_reason"]["maxLength"] == 4_000
    assert entity_schema["name"]["maxLength"] == 500
    assert entity_schema["evidence"]["maxLength"] == 2_000
    assert claim_schema["statement"]["maxLength"] == 2_000


def test_claude_provider_retries_one_schema_validation_failure() -> None:
    valid_output = EnrichmentOutput(
        tags=[
            TagAssignment(
                tag=TopicTag.PROMOTIONAL_SOCIAL,
                certainty=TagCertainty.CONFIDENT,
                source_refs=["tweet"],
            )
        ],
        information_status=InformationStatus.REPORTED,
        usefulness=Usefulness.IRRELEVANT,
        summary="Guardian Cap announced an endorsement with Jalen Pitre.",
        classification_reason="The source reports a commercial athlete endorsement.",
        entities=[],
        claims=[],
    )
    with pytest.raises(ValueError) as caught:
        EnrichmentOutput(
            tags=valid_output.tags,
            information_status=valid_output.information_status,
            usefulness=valid_output.usefulness,
            summary="x" * 2_001,
            classification_reason=valid_output.classification_reason,
        )

    calls: list[dict[str, object]] = []

    class Messages:
        def parse(self, **kwargs: object) -> object:
            calls.append(kwargs)
            if len(calls) == 1:
                raise caught.value
            return SimpleNamespace(
                parsed_output=valid_output,
                stop_reason="end_turn",
                usage=SimpleNamespace(
                    input_tokens=10,
                    output_tokens=5,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                ),
            )

    provider = ClaudeProvider.__new__(ClaudeProvider)
    provider.model_name = "test-model"
    provider.max_tokens = 1_000
    provider.client = SimpleNamespace(messages=Messages())

    response = provider.enrich(
        CollectedEvidence(text_sections=["[tweet]\nGuardian Cap inks Jalen Pitre."])
    )

    assert response.output == valid_output
    assert len(calls) == 2
    assert "failed output-schema validation" in str(calls[1]["system"])

    repaired = provider.repair(
        CollectedEvidence(text_sections=["[tweet]\nGuardian Cap inks Jalen Pitre."]),
        "Jalen Pitre: invalid excerpt",
    )

    assert repaired.output == valid_output
    assert len(calls) == 3
    assert "Do not combine separate passages" in str(calls[2]["system"])


def test_enrichment_settings_load_ignored_src_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("NEWS_ENRICHMENT_MODEL", raising=False)
    monkeypatch.delenv("NEWS_ENRICHMENT_VERSION", raising=False)
    monkeypatch.delenv("NEWS_ENRICHMENT_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("NEWS_ENRICHMENT_BATCH_SIZE", raising=False)
    (tmp_path / ".env").write_text(
        "ANTHROPIC_API_KEY=test-key\n"
        "NEWS_ENRICHMENT_MODEL=test-model\n"
        "NEWS_ENRICHMENT_VERSION=test-version\n"
        "NEWS_ENRICHMENT_MAX_OUTPUT_TOKENS=900\n"
        "NEWS_ENRICHMENT_BATCH_SIZE=2\n",
        encoding="utf-8",
    )

    settings = load_enrichment_settings(tmp_path)

    assert settings.api_key == "test-key"
    assert settings.model_name == "test-model"
    assert settings.enrichment_version == "test-version"
    assert settings.max_output_tokens == 900
    assert settings.batch_size == 2
    assert DEFAULT_MODEL_NAME == "claude-haiku-4-5-20251001"
    assert DEFAULT_MAX_OUTPUT_TOKENS == 1_536
    assert DEFAULT_BATCH_SIZE == 2


def test_enrichment_settings_reject_invalid_batch_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("NEWS_ENRICHMENT_BATCH_SIZE", "0")
    with pytest.raises(ValueError, match="NEWS_ENRICHMENT_BATCH_SIZE"):
        load_enrichment_settings(tmp_path)
    monkeypatch.setenv("NEWS_ENRICHMENT_BATCH_SIZE", "11")
    with pytest.raises(ValueError, match="NEWS_ENRICHMENT_BATCH_SIZE"):
        load_enrichment_settings(tmp_path)


def test_offline_pipeline_classifies_and_fingerprints() -> None:
    record = NewsRecord(
        news_id="x:1",
        text="Player left practice with an ankle injury and is questionable.",
        author_username="Reporter",
    )
    result = enrich_record(record, DeterministicDryRunProvider())
    assert result.status == "completed"
    assert result.output
    assert result.output.tags[0].tag == TopicTag.INJURY_AVAILABILITY
    assert result.output.usefulness == Usefulness.HIGH
    assert len(result.input_fingerprint) == 64
    assert result.provider == "deterministic_dry_run"


class _SourceRefProvider:
    provider_name = "test"
    model_name = "test-model"

    def __init__(self, source_ref: str) -> None:
        self.source_ref = source_ref

    def enrich(self, _evidence: object) -> ProviderResponse:
        output = EnrichmentOutput(
            tags=[
                TagAssignment(
                    tag=TopicTag.INJURY_AVAILABILITY,
                    certainty=TagCertainty.CONFIDENT,
                    source_refs=[self.source_ref],
                )
            ],
            information_status=InformationStatus.REPORTED,
            usefulness=Usefulness.HIGH,
            summary="A player is unavailable.",
            classification_reason="The post reports that the player is unavailable.",
        )
        return ProviderResponse(
            output=output,
            usage=ProviderUsage(input_tokens=12, output_tokens=5),
            model_name=self.model_name,
        )


class _EntityProvider:
    provider_name = "test"
    model_name = "test-model"

    def __init__(self, *, evidence: str) -> None:
        self.evidence = evidence

    def enrich(self, _evidence: object) -> ProviderResponse:
        output = EnrichmentOutput(
            tags=[
                TagAssignment(
                    tag=TopicTag.ROSTER_TRANSACTION,
                    certainty=TagCertainty.CONFIDENT,
                    source_refs=["article:test"],
                )
            ],
            information_status=InformationStatus.REPORTED,
            usefulness=Usefulness.HIGH,
            summary="Josh Allen remains with Buffalo.",
            classification_reason="The article names the player and team.",
            entities=[
                ExtractedEntity(
                    name="Josh Allen",
                    entity_type=EntityType.PLAYER,
                    mention_role=MentionRole.SUBJECT,
                    evidence=self.evidence,
                    confidence=0.9,
                    source_refs=["article:test"],
                ),
                ExtractedEntity(
                    name="Josh Allen",
                    entity_type=EntityType.PLAYER,
                    mention_role=MentionRole.SUBJECT,
                    evidence=self.evidence,
                    confidence=0.8,
                    source_refs=["article:test"],
                ),
            ],
        )
        return ProviderResponse(
            output=output,
            usage=ProviderUsage(input_tokens=20, output_tokens=10),
            model_name=self.model_name,
        )


def _output_with_entities(entities: list[ExtractedEntity]) -> EnrichmentOutput:
    return EnrichmentOutput(
        tags=[
            TagAssignment(
                tag=TopicTag.ROSTER_TRANSACTION,
                certainty=TagCertainty.CONFIDENT,
                source_refs=["tweet"],
            )
        ],
        information_status=InformationStatus.REPORTED,
        usefulness=Usefulness.HIGH,
        summary="Entity evidence test.",
        classification_reason="The source contains named NFL entities.",
        entities=entities,
        claims=[],
    )


class _EntityListProvider:
    provider_name = "test"
    model_name = "test-model"

    def __init__(self, entities: list[ExtractedEntity]) -> None:
        self.entities = entities

    def enrich(self, _evidence: object) -> ProviderResponse:
        return ProviderResponse(
            output=_output_with_entities(self.entities),
            usage=ProviderUsage(input_tokens=10, output_tokens=5),
            model_name=self.model_name,
        )


def test_enrichment_emits_deduplicated_resolution_ready_article_mentions() -> None:
    record = NewsRecord(
        news_id="x:article-mention",
        text="Read the linked roster report.",
        prepared_article_evidence=[
            {
                "source_ref": "article:test",
                "url": "https://example.test/roster",
                "title": "Roster report",
                "text": "Josh Allen remains with Buffalo.",
            }
        ],
    )

    result = enrich_record(
        record,
        _EntityProvider(evidence="Josh Allen remains with Buffalo."),
        enrichment_version="v3",
    )

    assert result.status == "completed"
    assert result.entity_extractor_version == ENTITY_EXTRACTOR_VERSION
    assert result.output is not None
    assert len(result.output.entities) == 1
    assert result.output.entities[0].mention_role == MentionRole.SUBJECT
    assert result.output.entities[0].source_refs == ["article:test"]


def test_enrichment_drops_entity_absent_from_collected_sources() -> None:
    record = NewsRecord(
        news_id="x:bad-evidence",
        text="No named player here.",
        prepared_article_evidence=[
            {
                "source_ref": "article:test",
                "url": "https://example.test/story",
                "text": "The article also contains no named player.",
            }
        ],
    )

    result = enrich_record(
        record,
        _EntityProvider(evidence="Josh Allen remains with Buffalo."),
        enrichment_version="v3",
    )

    assert result.status == "completed_with_warnings"
    assert result.error is None
    assert result.output is not None
    assert result.output.entities == []
    assert any("entity dropped after evidence validation" in item for item in result.warnings)


def test_enrichment_repairs_detached_excerpt_from_same_cited_source() -> None:
    record = NewsRecord(
        news_id="x:detached-evidence",
        text="Read the linked report.",
        prepared_article_evidence=[
            {
                "source_ref": "article:test",
                "url": "https://example.test/story",
                "text": "Josh Allen practiced. The Buffalo quarterback was limited.",
            }
        ],
    )

    result = enrich_record(
        record,
        _EntityProvider(evidence="The Buffalo quarterback was limited."),
        enrichment_version="v3",
    )

    assert result.status == "completed_with_warnings"
    assert result.output is not None
    assert "Josh Allen" in result.output.entities[0].evidence
    assert result.warnings == ["entity evidence repaired from cited source: Josh Allen"]


def test_enrichment_repairs_html_decoded_entity_evidence() -> None:
    record = NewsRecord(
        news_id="x:html-evidence",
        text="Hear from Josh Allen &amp; the Buffalo offense.",
    )
    provider = _EntityListProvider(
        [
            ExtractedEntity(
                name="Josh Allen",
                entity_type=EntityType.PLAYER,
                mention_role=MentionRole.SUBJECT,
                evidence="Hear from Josh Allen & the Buffalo offense.",
                confidence=0.9,
                source_refs=["tweet"],
            )
        ]
    )

    result = enrich_record(record, provider)

    assert result.status == "completed_with_warnings"
    assert result.output is not None
    assert result.output.entities[0].evidence == ("Hear from Josh Allen &amp; the Buffalo offense.")


def test_enrichment_repairs_paraphrased_excerpt_from_cited_article() -> None:
    record = NewsRecord(
        news_id="x:paraphrased-evidence",
        text="Read the linked report.",
        prepared_article_evidence=[
            {
                "source_ref": "article:test",
                "url": "https://example.test/story",
                "text": "Josh Allen practiced Tuesday. The quarterback was limited.",
            }
        ],
    )
    provider = _EntityListProvider(
        [
            ExtractedEntity(
                name="Josh Allen",
                entity_type=EntityType.PLAYER,
                mention_role=MentionRole.SUBJECT,
                evidence="The report says Josh Allen was limited in practice.",
                confidence=0.9,
                source_refs=["article:test"],
            )
        ]
    )

    result = enrich_record(record, provider)

    assert result.status == "completed_with_warnings"
    assert result.output is not None
    assert result.output.entities[0].evidence.startswith("Josh Allen practiced Tuesday.")


def test_enrichment_keeps_valid_entities_when_one_is_invalid() -> None:
    record = NewsRecord(news_id="x:partial-salvage", text="Josh Allen practiced.")
    provider = _EntityListProvider(
        [
            ExtractedEntity(
                name="Josh Allen",
                entity_type=EntityType.PLAYER,
                mention_role=MentionRole.SUBJECT,
                evidence="Josh Allen practiced.",
                confidence=0.9,
                source_refs=["tweet"],
            ),
            ExtractedEntity(
                name="Invented Player",
                entity_type=EntityType.PLAYER,
                mention_role=MentionRole.REFERENCED,
                evidence="Invented Player practiced.",
                confidence=0.6,
                source_refs=["tweet"],
            ),
        ]
    )

    result = enrich_record(record, provider)

    assert result.status == "completed_with_warnings"
    assert result.output is not None
    assert [entity.name for entity in result.output.entities] == ["Josh Allen"]
    assert any("Invented Player" in item for item in result.warnings)


def test_enrichment_retries_unrepairable_entity_evidence_once() -> None:
    record = NewsRecord(
        news_id="x:provider-repair",
        text="Josh Allen practiced.",
        prepared_media_evidence=[
            {
                "source_ref": "media:test",
                "media_type": "image",
                "ocr_text": "Practice report",
            }
        ],
    )
    invalid = ExtractedEntity(
        name="Josh Allen",
        entity_type=EntityType.PLAYER,
        mention_role=MentionRole.SUBJECT,
        evidence="Josh Allen was at practice.",
        confidence=0.9,
        source_refs=["media:test"],
    )
    valid = invalid.model_copy(
        update={"evidence": "Josh Allen practiced.", "source_refs": ["tweet"]}
    )

    class RepairingProvider(_EntityListProvider):
        repair_calls = 0

        def repair(
            self,
            _evidence: object,
            validation_feedback: str,
        ) -> ProviderResponse:
            self.repair_calls += 1
            assert "no verbatim excerpt" in validation_feedback
            return ProviderResponse(
                output=_output_with_entities([valid]),
                usage=ProviderUsage(input_tokens=7, output_tokens=3),
                model_name=self.model_name,
            )

    provider = RepairingProvider([invalid])
    result = enrich_record(record, provider)

    assert result.status == "completed_with_warnings"
    assert result.output is not None
    assert result.output.entities[0].evidence == "Josh Allen practiced."
    assert provider.repair_calls == 1
    assert result.usage.input_tokens == 17
    assert result.usage.output_tokens == 8


def test_enrichment_prompt_excludes_brand_signings_from_roster_transactions() -> None:
    assert "exclude endorsements, sponsorships" in SYSTEM_PROMPT
    assert "brand, equipment vendor" in SYSTEM_PROMPT


def test_pipeline_normalizes_exact_tweet_url_source_reference() -> None:
    source_url = "https://x.com/reporter/status/123"
    record = NewsRecord(
        news_id="x:123",
        text="The player is out.",
        source_url=source_url,
    )

    result = enrich_record(record, _SourceRefProvider(source_url))

    assert result.status == "completed"
    assert result.output
    assert result.output.tags[0].source_refs == ["tweet"]


def test_pipeline_rejects_unknown_reference_and_preserves_usage() -> None:
    record = NewsRecord(news_id="x:124", text="The player is out.")

    result = enrich_record(record, _SourceRefProvider("https://invented.example/source"))

    assert result.status == "failed"
    assert result.error
    assert "unknown source references" in result.error
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 5


def test_network_disabled_records_partial_article_and_media_status() -> None:
    record = NewsRecord(
        news_id="x:2",
        text="Story and image attached.",
        source_entities={
            "urls": [{"expanded_url": "https://example.com/story"}],
        },
        media=[
            {
                "media_key": "image-1",
                "media_type": "photo",
                "source_url": "https://example.com/image.jpg",
            }
        ],
    )
    evidence = collect_evidence(record, allow_network=False)
    assert evidence.manifest["articles"][0]["status"] == "network_disabled"
    assert evidence.manifest["media"][0]["status"] == "network_disabled"
    assert len(evidence.warnings) == 2
    assert "article:1" not in evidence.source_refs()
    assert "media:image-1" in evidence.source_refs()


def test_prepared_evidence_is_available_offline() -> None:
    record = NewsRecord(
        news_id="x:3",
        text="Practice report attached.",
        prepared_media_evidence=[
            {
                "source_ref": "media:test",
                "media_type": "image",
                "ocr_text": "Player Example — Did Not Practice — Ankle",
            }
        ],
        prepared_article_evidence=[
            {
                "source_ref": "article:test",
                "url": "https://example.com/story",
                "title": "Practice update",
                "text": "Player Example missed practice.",
            }
        ],
    )
    evidence = collect_evidence(record)
    prompt = evidence.as_prompt_text()
    assert "Did Not Practice" in prompt
    assert "Player Example missed practice" in prompt
    assert not evidence.warnings


def test_article_url_selection_deduplicates_and_excludes_x() -> None:
    urls = article_urls(
        {
            "urls": [
                {"expanded_url": "https://x.com/team/status/1"},
                {"expanded_url": "https://example.com/story"},
                {"unwound_url": "https://example.com/story"},
            ]
        }
    )
    assert urls == ["https://example.com/story"]


def test_public_url_validation_rejects_private_resolution() -> None:
    def private_resolver(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    def public_resolver(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    try:
        validate_public_url("https://internal.example/test", resolver=private_resolver)
    except ValueError as exc:
        assert "private or reserved" in str(exc)
    else:
        raise AssertionError("private resolution should be rejected")
    assert (
        validate_public_url("https://public.example/test", resolver=public_resolver)
        == "https://public.example/test"
    )


def test_article_extraction_removes_navigation_and_scripts() -> None:
    resource = DownloadedResource(
        final_url="https://example.com/story",
        content_type="text/html",
        body=(
            b"<html><head><title>Roster move</title><script>bad()</script></head>"
            b"<body><nav>Menu</nav><article><h1>Trade</h1>"
            b"<p>The team acquired a receiver.</p></article></body></html>"
        ),
    )
    title, text = extract_article_text(resource)
    assert title == "Roster move"
    assert "acquired a receiver" in text
    assert "Menu" not in text
    assert "bad()" not in text


def test_local_image_is_normalized_for_model(tmp_path: Path) -> None:
    image_path = tmp_path / "report.png"
    Image.new("RGB", (2000, 1000), color="white").save(image_path)
    record = NewsRecord(
        news_id="x:4",
        text="Report attached.",
        media=[
            {
                "media_key": "image-2",
                "media_type": "photo",
                "local_path": str(image_path),
            }
        ],
    )
    evidence = collect_evidence(record)
    assert len(evidence.images) == 1
    assert evidence.images[0].media_type == "image/jpeg"
    assert evidence.manifest["media"][0]["status"] == "image_prepared"


def test_local_video_is_sampled_into_model_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "update.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x180:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(video_path),
        ],
        check=True,
        timeout=30,
    )
    record = NewsRecord(
        news_id="x:video",
        text="Coach update in the attached video.",
        media=[
            {
                "media_key": "video-1",
                "media_type": "video",
                "local_path": str(video_path),
            }
        ],
    )
    evidence = collect_evidence(record)
    assert evidence.images
    assert evidence.images[0].source_ref.startswith("media:video-1:frame:")
    assert evidence.manifest["media"][0]["status"] == "frames_extracted"
    assert evidence.manifest["media"][0]["audio_status"] == "not_transcribed"


def test_video_sample_timestamps_stay_inside_duration() -> None:
    timestamps = _video_sample_timestamps(90.0)

    assert timestamps[0] == 0.0
    assert timestamps[-1] < 90.0
    assert len(timestamps) == 3


def test_repository_values_keep_tags_queryable() -> None:
    output = EnrichmentOutput(
        tags=[
            TagAssignment(
                tag=TopicTag.WEATHER_FIELD_CONDITIONS,
                certainty=TagCertainty.CONFIDENT,
                source_refs=["tweet"],
            )
        ],
        information_status=InformationStatus.REPORTED,
        usefulness=Usefulness.HIGH,
        summary="High wind is expected.",
        classification_reason="The post reports game-time weather.",
    )
    result = EnrichmentResult(
        news_id="x:5",
        enrichment_version="v1",
        entity_extractor_version="news-enrichment-mentions-v2",
        provider="anthropic",
        model_name=DEFAULT_MODEL_NAME,
        status="completed",
        input_fingerprint="a" * 64,
        input_manifest={},
        output=output,
        usage=ProviderUsage(input_tokens=100, output_tokens=50),
    )
    row = enrichment_values(result)
    tags = tag_values(result)
    assert "primary_tag" not in row
    assert "confidence" not in row
    assert row["usage"]["input_tokens"] == 100
    assert tags[0]["tag"] == "weather_field_conditions"
    assert tags[0]["certainty"] == "confident"
    assert "confidence" not in tags[0]


def test_worker_rejects_persisting_mock_results(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = worker_main(
        [
            "--provider",
            "mock",
            "--apply",
            "--confirm-live-writes",
            "APPLY_ENRICHMENTS",
        ]
    )

    assert exit_code == 2
    assert "--apply requires --provider claude" in capsys.readouterr().err


def test_dry_run_cli_writes_only_local_results(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "news_id": "x:6",
                "text": "The player was ruled out with a knee injury.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    exit_code = dry_run_main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--provider",
            "mock",
        ]
    )
    assert exit_code == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert summary["durable_external_writes"] is False
    result = EnrichmentResult.model_validate_json(
        (output_dir / "enrichment_results.jsonl").read_text(encoding="utf-8")
    )
    assert result.output
    assert result.output.tags[0].tag == TopicTag.INJURY_AVAILABILITY


def _fixture_output(*, source_ref: str = "tweet", summary: str = "Injury update.") -> EnrichmentOutput:
    return EnrichmentOutput(
        tags=[
            TagAssignment(
                tag=TopicTag.INJURY_AVAILABILITY,
                certainty=TagCertainty.CONFIDENT,
                source_refs=[source_ref],
            )
        ],
        information_status=InformationStatus.REPORTED,
        usefulness=Usefulness.HIGH,
        summary=summary,
        classification_reason="The source reports an injury.",
    )


def test_split_usage_preserves_token_totals() -> None:
    usage = ProviderUsage(input_tokens=11, output_tokens=5)
    shares = split_usage(usage, 2)
    assert shares[0].input_tokens + shares[1].input_tokens == 11
    assert shares[0].output_tokens + shares[1].output_tokens == 5


def test_enrich_records_uses_one_batch_call() -> None:
    calls: list[list[str]] = []

    class BatchProvider:
        provider_name = "batch"
        model_name = "test-model"

        def enrich(self, _evidence: object) -> ProviderResponse:
            raise AssertionError("single-tweet enrich should not be used for a pair")

        def enrich_many(
            self,
            items: list[tuple[str, object]],
        ) -> dict[str, ProviderResponse]:
            calls.append([news_id for news_id, _evidence in items])
            return {
                news_id: ProviderResponse(
                    output=_fixture_output(summary=f"Summary for {news_id}."),
                    usage=ProviderUsage(input_tokens=3, output_tokens=1),
                    model_name=self.model_name,
                )
                for news_id, _evidence in items
            }

    results = enrich_records(
        [
            NewsRecord(news_id="x:1", text="Player left practice with an ankle injury."),
            NewsRecord(news_id="x:2", text="Starter is questionable for Sunday."),
        ],
        BatchProvider(),
    )

    assert calls == [["x:1", "x:2"]]
    assert [result.news_id for result in results] == ["x:1", "x:2"]
    assert all(result.status == "completed" for result in results)
    assert results[0].usage.input_tokens == 3


def test_enrich_records_fails_missing_news_id_only() -> None:
    class PartialProvider:
        provider_name = "batch"
        model_name = "test-model"

        def enrich(self, _evidence: object) -> ProviderResponse:
            raise AssertionError("unused")

        def enrich_many(
            self,
            items: list[tuple[str, object]],
        ) -> dict[str, ProviderResponse]:
            news_id, _evidence = items[0]
            return {
                news_id: ProviderResponse(
                    output=_fixture_output(),
                    usage=ProviderUsage(),
                    model_name=self.model_name,
                )
            }

    results = enrich_records(
        [
            NewsRecord(news_id="x:1", text="Player left practice with an ankle injury."),
            NewsRecord(news_id="x:2", text="Starter is questionable for Sunday."),
        ],
        PartialProvider(),
    )

    assert results[0].status == "completed"
    assert results[1].status == "failed"
    assert results[1].error
    assert "x:2" in results[1].error


def test_claude_provider_enrich_many_namespaces_and_splits_usage() -> None:
    parsed = BatchedEnrichmentOutput(
        tweets=[
            PerTweetEnrichment(
                news_id="x:1",
                tags=[
                    TagAssignment(
                        tag=TopicTag.INJURY_AVAILABILITY,
                        certainty=TagCertainty.CONFIDENT,
                        source_refs=["x:1/tweet"],
                    )
                ],
                information_status=InformationStatus.REPORTED,
                usefulness=Usefulness.HIGH,
                summary="Tweet one injury.",
                classification_reason="The first tweet reports an injury.",
            ),
            PerTweetEnrichment(
                news_id="x:2",
                tags=[
                    TagAssignment(
                        tag=TopicTag.INJURY_AVAILABILITY,
                        certainty=TagCertainty.CONFIDENT,
                        source_refs=["x:2/tweet"],
                    )
                ],
                information_status=InformationStatus.REPORTED,
                usefulness=Usefulness.HIGH,
                summary="Tweet two injury.",
                classification_reason="The second tweet reports an injury.",
            ),
        ]
    )
    calls: list[dict[str, object]] = []

    class Messages:
        def parse(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(
                parsed_output=parsed,
                stop_reason="end_turn",
                usage=SimpleNamespace(
                    input_tokens=11,
                    output_tokens=5,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                ),
            )

    provider = ClaudeProvider.__new__(ClaudeProvider)
    provider.model_name = "test-model"
    provider.max_tokens = 1_536
    provider.client = SimpleNamespace(messages=Messages())

    responses = provider.enrich_many(
        [
            ("x:1", CollectedEvidence(text_sections=["[tweet]\nPlayer left practice."])),
            ("x:2", CollectedEvidence(text_sections=["[tweet]\nStarter is questionable."])),
        ]
    )

    assert set(responses) == {"x:1", "x:2"}
    assert responses["x:1"].output.tags[0].source_refs == ["tweet"]
    assert responses["x:2"].output.tags[0].source_refs == ["tweet"]
    assert responses["x:1"].usage.input_tokens + responses["x:2"].usage.input_tokens == 11
    assert calls[0]["output_format"] is BatchedEnrichmentOutput
    assert int(calls[0]["max_tokens"]) >= 4_096
    assert "independent tweets" in str(calls[0]["messages"])
