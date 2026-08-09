"""Fetch and normalize current NFL coaching staffs from official club sites."""

from __future__ import annotations

import base64
import gzip
import hashlib
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag

from src.common.gcs import canonical_json_bytes
from src.entity_bank.models import AliasType, EntityType, PersonRoleHint
from src.entity_bank.nflverse_pipeline import ENTITY_NAMESPACE
from src.entity_bank.normalization import normalize_name

SCHEMA_NAME = "nfl_coach_entity_snapshot"
SCHEMA_VERSION = 1
STORAGE_PROVIDER = "nfl"
STORAGE_SOURCE = "club-sites"
STORAGE_OBJECT = "coach-entity-snapshot"
BANK_SOURCE = "nfl_club_sites"
COACH_NORMALIZER_VERSION = "nfl-club-coach-normalizer-v1"
MINIMUM_STAFF_PER_TEAM = 10


@dataclass(frozen=True)
class TeamCoachSource:
    team_id: str
    team_name: str
    url: str


TEAM_COACH_SOURCES: dict[str, TeamCoachSource] = {
    "ARI": TeamCoachSource(
        "3800", "Arizona Cardinals", "https://www.azcardinals.com/team/coaches-roster/"
    ),
    "ATL": TeamCoachSource(
        "0200", "Atlanta Falcons", "https://www.atlantafalcons.com/team/coaches-roster/"
    ),
    "BAL": TeamCoachSource(
        "0325", "Baltimore Ravens", "https://www.baltimoreravens.com/team/coaches-roster/"
    ),
    "BUF": TeamCoachSource(
        "0610", "Buffalo Bills", "https://www.buffalobills.com/team/coaches-roster/"
    ),
    "CAR": TeamCoachSource(
        "0750", "Carolina Panthers", "https://www.panthers.com/team/coaches-roster/"
    ),
    "CHI": TeamCoachSource("0810", "Chicago Bears", "https://www.chicagobears.com/team/coaches/"),
    "CIN": TeamCoachSource(
        "0920", "Cincinnati Bengals", "https://www.bengals.com/team/coaches-roster/"
    ),
    "CLE": TeamCoachSource(
        "1050", "Cleveland Browns", "https://www.clevelandbrowns.com/team/coaches-roster/"
    ),
    "DAL": TeamCoachSource(
        "1200", "Dallas Cowboys", "https://www.dallascowboys.com/team/coaches-roster/"
    ),
    "DEN": TeamCoachSource(
        "1400", "Denver Broncos", "https://www.denverbroncos.com/team/coaches-roster/"
    ),
    "DET": TeamCoachSource(
        "1540", "Detroit Lions", "https://www.detroitlions.com/team/coaches-roster/"
    ),
    "GB": TeamCoachSource(
        "1800", "Green Bay Packers", "https://www.packers.com/team/coaches-roster/"
    ),
    "HOU": TeamCoachSource(
        "2120", "Houston Texans", "https://www.houstontexans.com/team/coaches-roster/"
    ),
    "IND": TeamCoachSource(
        "2200", "Indianapolis Colts", "https://www.colts.com/team/coaches-roster/"
    ),
    "JAX": TeamCoachSource(
        "2250", "Jacksonville Jaguars", "https://www.jaguars.com/team/coaches-roster/"
    ),
    "KC": TeamCoachSource(
        "2310", "Kansas City Chiefs", "https://www.chiefs.com/team/coaches-roster/"
    ),
    "LAC": TeamCoachSource(
        "4400", "Los Angeles Chargers", "https://www.chargers.com/team/coaches-roster/"
    ),
    "LAR": TeamCoachSource(
        "2510", "Los Angeles Rams", "https://www.therams.com/team/coaches-roster/"
    ),
    "LV": TeamCoachSource(
        "2520", "Las Vegas Raiders", "https://www.raiders.com/team/coaches-roster/"
    ),
    "MIA": TeamCoachSource(
        "2700", "Miami Dolphins", "https://www.miamidolphins.com/team/coaches-roster/"
    ),
    "MIN": TeamCoachSource(
        "3000", "Minnesota Vikings", "https://www.vikings.com/team/coaches-roster/"
    ),
    "NE": TeamCoachSource(
        "3200", "New England Patriots", "https://www.patriots.com/team/coaches-roster/"
    ),
    "NO": TeamCoachSource(
        "3300", "New Orleans Saints", "https://www.neworleanssaints.com/team/coaches-roster/"
    ),
    "NYG": TeamCoachSource(
        "3410", "New York Giants", "https://www.giants.com/team/coaches-roster/"
    ),
    "NYJ": TeamCoachSource(
        "3430", "New York Jets", "https://www.newyorkjets.com/team/coaches-roster/"
    ),
    "PHI": TeamCoachSource(
        "3700", "Philadelphia Eagles", "https://www.philadelphiaeagles.com/team/coaches/"
    ),
    "PIT": TeamCoachSource(
        "3900", "Pittsburgh Steelers", "https://www.steelers.com/team/coaches-roster/"
    ),
    "SEA": TeamCoachSource(
        "4600", "Seattle Seahawks", "https://www.seahawks.com/team/coaches-roster/"
    ),
    "SF": TeamCoachSource(
        "4500", "San Francisco 49ers", "https://www.49ers.com/team/coaches-roster/"
    ),
    "TB": TeamCoachSource(
        "4900", "Tampa Bay Buccaneers", "https://www.buccaneers.com/team/coaches-roster/"
    ),
    "TEN": TeamCoachSource(
        "2100", "Tennessee Titans", "https://www.tennesseetitans.com/team/coaches-roster/"
    ),
    "WAS": TeamCoachSource(
        "5110", "Washington Commanders", "https://www.commanders.com/team/coaches-roster/"
    ),
}


@dataclass(frozen=True)
class CoachPageAsset:
    team_abbreviation: str
    url: str
    content: bytes
    etag: str | None
    sha256: str
    source_updated_at: str | None = None


@dataclass(frozen=True)
class CoachSnapshot:
    season: int
    observed_at: datetime
    assets: tuple[CoachPageAsset, ...]
    entities: tuple[dict[str, Any], ...]
    relationships: tuple[dict[str, Any], ...]
    quarantined_records: tuple[dict[str, Any], ...]
    quality: dict[str, int]
    content_sha256: str


class OfficialCoachClient:
    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        timeout_seconds: float = 60,
        max_attempts: int = 4,
        sleep: Any = time.sleep,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.sleep = sleep

    def _get(self, url: str) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.session.get(
                    url,
                    timeout=self.timeout_seconds,
                    headers={"User-Agent": "AI-SportsBettor entity-bank sync/1.0"},
                )
                response.raise_for_status()
                return response
            except (requests.RequestException, RuntimeError) as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    self.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"official NFL club request failed: {url}") from last_error

    def fetch(self) -> tuple[CoachPageAsset, ...]:
        assets = []
        for team_abbreviation, source in sorted(TEAM_COACH_SOURCES.items()):
            response = self._get(source.url)
            content = response.content
            assets.append(
                CoachPageAsset(
                    team_abbreviation=team_abbreviation,
                    url=response.url,
                    content=content,
                    etag=response.headers.get("ETag"),
                    sha256=hashlib.sha256(content).hexdigest(),
                    source_updated_at=response.headers.get("Last-Modified"),
                )
            )
        return tuple(assets)


def _clean_text(node: Tag | None) -> str:
    if node is None:
        return ""
    return " ".join(node.get_text(" ", strip=True).split())


def _person_card(node: Tag) -> Tag | None:
    parent: Tag | None = node
    while parent is not None:
        if any("person-card" in class_name for class_name in parent.get("class", [])):
            return parent
        parent = parent.parent if isinstance(parent.parent, Tag) else None
    return None


def parse_coach_page(asset: CoachPageAsset) -> list[dict[str, str | None]]:
    """Parse only person cards, excluding similarly styled news/content cards."""

    soup = BeautifulSoup(asset.content, "html.parser")
    records: dict[tuple[str, str, str | None], dict[str, str | None]] = {}
    for name_node in soup.select("h3.d3-o-media-object__title"):
        card = _person_card(name_node)
        if card is None:
            continue
        title_node = card.select_one(".d3-o-media-object__roofline")
        canonical_name = _clean_text(name_node)
        staff_title = _clean_text(title_node)
        if not canonical_name or not staff_title:
            continue
        link = card if card.name == "a" and card.get("href") else card.find("a", href=True)
        profile_url = urljoin(asset.url, str(link["href"])) if link is not None else None
        key = (normalize_name(canonical_name), staff_title.casefold(), profile_url)
        records[key] = {
            "canonical_name": canonical_name,
            "staff_title": staff_title,
            "profile_url": profile_url,
        }
    return list(records.values())


def _team_entity_id(team_id: str) -> str:
    return str(uuid.uuid5(ENTITY_NAMESPACE, f"nflverse:team:{team_id}"))


def _coach_entity_id(source_id: str) -> str:
    return str(uuid.uuid5(ENTITY_NAMESPACE, f"{BANK_SOURCE}:person:{source_id}"))


def normalize_coach_snapshot(
    assets: tuple[CoachPageAsset, ...],
    *,
    season: int,
    observed_at: datetime,
) -> CoachSnapshot:
    parsed_by_team: dict[str, list[dict[str, str | None]]] = {}
    quarantined: list[dict[str, Any]] = []
    for asset in assets:
        if asset.team_abbreviation not in TEAM_COACH_SOURCES:
            raise ValueError(f"unknown NFL team abbreviation {asset.team_abbreviation!r}")
        if asset.team_abbreviation in parsed_by_team:
            raise ValueError(f"duplicate coach page for {asset.team_abbreviation}")
        records = parse_coach_page(asset)
        parsed_by_team[asset.team_abbreviation] = records

    incomplete_teams = {
        team: len(records)
        for team, records in parsed_by_team.items()
        if len(records) < MINIMUM_STAFF_PER_TEAM
    }
    missing_head_coaches = sorted(
        team
        for team, records in parsed_by_team.items()
        if sum(str(record["staff_title"]).casefold() == "head coach" for record in records) != 1
    )
    if incomplete_teams or missing_head_coaches:
        raise ValueError(
            "official coach pages failed completeness checks: "
            f"short_staffs={incomplete_teams}, missing_or_duplicate_head_coaches="
            f"{missing_head_coaches}"
        )

    records_by_name: dict[str, list[dict[str, Any]]] = {}
    for team_abbreviation, records in parsed_by_team.items():
        for record in records:
            normalized_name = normalize_name(str(record["canonical_name"]))
            if not normalized_name:
                quarantined.append(
                    {
                        "source": BANK_SOURCE,
                        "status": "quarantined",
                        "reason_codes": ["missing_person_name"],
                        "record": {"team": team_abbreviation, **record},
                    }
                )
                continue
            records_by_name.setdefault(normalized_name, []).append(
                {"team_abbreviation": team_abbreviation, **record}
            )

    conflicting_names = {
        normalized_name
        for normalized_name, records in records_by_name.items()
        if len({record["team_abbreviation"] for record in records}) > 1
    }
    for normalized_name in sorted(conflicting_names):
        quarantined.append(
            {
                "source": BANK_SOURCE,
                "status": "excluded_from_bank",
                "reason_codes": ["normalized_name_listed_by_multiple_teams"],
                "normalized_name": normalized_name,
                "records": records_by_name[normalized_name],
            }
        )

    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    for normalized_name, records in sorted(records_by_name.items()):
        if normalized_name in conflicting_names:
            continue
        team_abbreviation = str(records[0]["team_abbreviation"])
        team = TEAM_COACH_SOURCES[team_abbreviation]
        display_names = sorted({str(record["canonical_name"]) for record in records})
        canonical_name = max(display_names, key=lambda value: (len(value), value))
        staff_titles = sorted({str(record["staff_title"]) for record in records})
        profile_urls = sorted(
            {str(record["profile_url"]) for record in records if record.get("profile_url")}
        )
        source_id = normalized_name
        if len(source_id) > 128:
            quarantined.append(
                {
                    "source": BANK_SOURCE,
                    "status": "quarantined",
                    "reason_codes": ["source_identifier_too_long"],
                    "records": records,
                }
            )
            continue
        entity_id = _coach_entity_id(source_id)
        aliases = [
            {
                "alias": display_name,
                "normalized_alias": normalize_name(display_name),
                "alias_type": (
                    AliasType.CANONICAL_NAME.value
                    if display_name == canonical_name
                    else AliasType.PROVIDER_NAME.value
                ),
            }
            for display_name in display_names
        ]
        evidence = {
            "team_abbreviation": team_abbreviation,
            "team_name": team.team_name,
            "staff_titles": staff_titles,
            "profile_urls": profile_urls,
            "directory_url": team.url,
        }
        entities.append(
            {
                "entity_id": entity_id,
                "entity_type": EntityType.PERSON.value,
                "canonical_name": canonical_name,
                "normalized_name": normalized_name,
                "source_mappings": [
                    {
                        "provider": BANK_SOURCE,
                        "source_entity_type": "person",
                        "source_entity_id": source_id,
                        "metadata": evidence,
                    }
                ],
                "aliases": aliases,
                "roles": [
                    {
                        "role": PersonRoleHint.COACH.value,
                        "source": BANK_SOURCE,
                        "evidence": evidence,
                    }
                ],
            }
        )
        source_key = f"{BANK_SOURCE}:coach:{season}:{source_id}:{team_abbreviation}"
        relationships.append(
            {
                "relationship_id": str(uuid.uuid5(ENTITY_NAMESPACE, source_key)),
                "subject_entity_id": entity_id,
                "predicate": "coaches_for",
                "object_entity_id": _team_entity_id(team.team_id),
                "source": BANK_SOURCE,
                "source_key": source_key,
                "evidence": evidence,
            }
        )

    content_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "normalizer_version": COACH_NORMALIZER_VERSION,
                "assets": {asset.team_abbreviation: asset.sha256 for asset in assets},
            }
        )
    ).hexdigest()
    return CoachSnapshot(
        season=season,
        observed_at=observed_at.astimezone(UTC),
        assets=assets,
        entities=tuple(entities),
        relationships=tuple(relationships),
        quarantined_records=tuple(quarantined),
        quality={
            "source_team_pages": len(assets),
            "source_staff_records": sum(len(records) for records in parsed_by_team.values()),
            "canonical_coaches": len(entities),
            "quarantined_records": len(quarantined),
            "unsafe_source_mapping_collisions": len(conflicting_names),
        },
        content_sha256=content_sha256,
    )


def fetch_coach_snapshot(
    client: OfficialCoachClient,
    *,
    season: int,
    now: datetime | None = None,
) -> CoachSnapshot:
    observed_at = (now or datetime.now(UTC)).astimezone(UTC)
    return normalize_coach_snapshot(
        client.fetch(),
        season=season,
        observed_at=observed_at,
    )


def build_coach_object_path(observed_at: datetime, ingest_run_id: str) -> str:
    utc = observed_at.astimezone(UTC)
    return (
        f"raw/provider={STORAGE_PROVIDER}/source={STORAGE_SOURCE}/"
        f"object={STORAGE_OBJECT}/schema=v{SCHEMA_VERSION}/"
        f"date={utc:%Y-%m-%d}/hour={utc:%H}/"
        f"nfl_coaches_{ingest_run_id}.json.gz"
    )


def build_coach_envelope(
    snapshot: CoachSnapshot,
    *,
    ingest_run_id: str,
    storage_uri: str,
) -> dict[str, Any]:
    asset_metadata = [
        {
            "team_abbreviation": asset.team_abbreviation,
            "url": asset.url,
            "etag": asset.etag,
            "sha256": asset.sha256,
            "source_updated_at": asset.source_updated_at,
            "byte_size": len(asset.content),
        }
        for asset in snapshot.assets
    ]
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "provider": STORAGE_PROVIDER,
        "source": STORAGE_SOURCE,
        "object_type": STORAGE_OBJECT,
        "ingest_run_id": ingest_run_id,
        "ingested_at": snapshot.observed_at.isoformat(),
        "storage_uri": storage_uri,
        "content_sha256": snapshot.content_sha256,
        "record_count": len(snapshot.entities) + len(snapshot.relationships),
        "request": {
            "season": snapshot.season,
            "normalizer_version": COACH_NORMALIZER_VERSION,
            "assets": asset_metadata,
        },
        "snapshot": {
            "season": snapshot.season,
            "entity_count": len(snapshot.entities),
            "relationship_count": len(snapshot.relationships),
            "normalization_audit": {
                "quality": snapshot.quality,
                "quarantined_records": list(snapshot.quarantined_records),
            },
            "assets": [
                {
                    **metadata,
                    "content_base64": base64.b64encode(asset.content).decode("ascii"),
                }
                for metadata, asset in zip(asset_metadata, snapshot.assets, strict=True)
            ],
        },
        "_normalized_entities": list(snapshot.entities),
        "_normalized_relationships": list(snapshot.relationships),
    }


def archive_coach_envelope(envelope: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in envelope.items() if not key.startswith("_normalized_")}


def encode_coach_envelope(envelope: dict[str, Any]) -> bytes:
    return gzip.compress(
        canonical_json_bytes(archive_coach_envelope(envelope)),
        compresslevel=6,
        mtime=0,
    )


def coach_summary(snapshot: CoachSnapshot) -> dict[str, Any]:
    return {
        "season": snapshot.season,
        "normalizer_version": COACH_NORMALIZER_VERSION,
        "content_sha256": snapshot.content_sha256,
        "coaches": len(snapshot.entities),
        "relationships": len(snapshot.relationships),
        "quality": snapshot.quality,
        "assets": [
            {
                "team_abbreviation": asset.team_abbreviation,
                "sha256": asset.sha256,
                "source_updated_at": asset.source_updated_at,
                "byte_size": len(asset.content),
            }
            for asset in snapshot.assets
        ],
    }
