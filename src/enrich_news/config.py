"""Configuration shared by local enrichment entry points."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

DEFAULT_MODEL_NAME = "claude-haiku-4-5-20251001"
DEFAULT_ENRICHMENT_VERSION = "v3"
DEFAULT_MAX_OUTPUT_TOKENS = 1_536
MIN_OUTPUT_TOKENS = 256
MAX_OUTPUT_TOKENS = 4_096
DEFAULT_BATCH_SIZE = 2
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 10


@dataclass(frozen=True)
class EnrichmentSettings:
    api_key: str | None
    model_name: str
    enrichment_version: str
    max_output_tokens: int
    batch_size: int = DEFAULT_BATCH_SIZE


def load_enrichment_settings(src_dir: Path) -> EnrichmentSettings:
    """Load process overrides first, followed by the ignored ``src/.env`` file."""

    dotenv_path = src_dir / ".env"
    file_values = dotenv_values(dotenv_path) if dotenv_path.exists() else {}
    values = {key: str(value) for key, value in file_values.items() if value is not None}
    values.update(os.environ)

    max_output_tokens = int(
        values.get("NEWS_ENRICHMENT_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)
    )
    if not MIN_OUTPUT_TOKENS <= max_output_tokens <= MAX_OUTPUT_TOKENS:
        raise ValueError(
            "NEWS_ENRICHMENT_MAX_OUTPUT_TOKENS must be between "
            f"{MIN_OUTPUT_TOKENS} and {MAX_OUTPUT_TOKENS}"
        )
    batch_size = int(values.get("NEWS_ENRICHMENT_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    if not MIN_BATCH_SIZE <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(
            "NEWS_ENRICHMENT_BATCH_SIZE must be between "
            f"{MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}"
        )
    return EnrichmentSettings(
        api_key=values.get("ANTHROPIC_API_KEY"),
        model_name=values.get("NEWS_ENRICHMENT_MODEL", DEFAULT_MODEL_NAME),
        enrichment_version=values.get(
            "NEWS_ENRICHMENT_VERSION",
            DEFAULT_ENRICHMENT_VERSION,
        ),
        max_output_tokens=max_output_tokens,
        batch_size=batch_size,
    )
