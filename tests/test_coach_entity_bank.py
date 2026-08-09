from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from src.entity_bank.coach_pipeline import (
    BANK_SOURCE,
    CoachPageAsset,
    normalize_coach_snapshot,
    parse_coach_page,
)
from src.entity_bank.coach_sync import main as coach_sync_main

NOW = datetime(2026, 8, 9, 12, tzinfo=UTC)


def coach_asset(team: str, records: list[tuple[str, str]]) -> CoachPageAsset:
    cards = "".join(
        f"""
        <a class="d3-o-media-object d3-o-person-card--non-featured-coach"
           href="/team/coaches-roster/{name.lower().replace(" ", "-")}">
          <div class="d3-o-media-object__body">
            <h3 class="d3-o-media-object__title">{name}</h3>
            <h5 class="d3-o-media-object__roofline">{title}</h5>
          </div>
        </a>
        """
        for name, title in records
    )
    news_pollution = """
      <div class="d3-o-media-object d3-o-content-tray__card">
        <h3 class="d3-o-media-object__title">Coach rumor news headline</h3>
        <h5 class="d3-o-media-object__roofline">news</h5>
      </div>
    """
    content = f"<html><body>{news_pollution}{cards}</body></html>".encode()
    return CoachPageAsset(
        team_abbreviation=team,
        url=f"https://example.test/{team.lower()}/team/coaches-roster/",
        content=content,
        etag='"fixture"',
        sha256=hashlib.sha256(content).hexdigest(),
    )


def staff(prefix: str, *, head_coach: str) -> list[tuple[str, str]]:
    return [
        (head_coach, "Head Coach"),
        (f"{prefix} Coordinator", "Offensive Coordinator"),
        (f"{prefix} Defense", "Defensive Coordinator"),
        (f"{prefix} Quarterbacks", "Quarterbacks Coach"),
        (f"{prefix} Receivers", "Wide Receivers"),
        (f"{prefix} Backs", "Running Backs"),
        (f"{prefix} Line", "Offensive Line"),
        (f"{prefix} Secondary", "Secondary"),
        (f"{prefix} Teams", "Special Teams Assistant"),
        (f"{prefix} Strength", "Strength and Conditioning"),
    ]


def test_coach_page_parser_keeps_person_cards_and_excludes_news() -> None:
    asset = coach_asset("BUF", staff("Buffalo", head_coach="Joe Brady"))

    records = parse_coach_page(asset)

    assert len(records) == 10
    assert records[0] == {
        "canonical_name": "Joe Brady",
        "staff_title": "Head Coach",
        "profile_url": "https://example.test/team/coaches-roster/joe-brady",
    }
    assert all(record["staff_title"] != "news" for record in records)


def test_coach_snapshot_builds_canonical_people_roles_and_team_relationships() -> None:
    snapshot = normalize_coach_snapshot(
        (
            coach_asset("BUF", staff("Buffalo", head_coach="Joe Brady")),
            coach_asset("CIN", staff("Cincinnati", head_coach="Zac Taylor")),
        ),
        season=2026,
        observed_at=NOW,
    )

    assert len(snapshot.entities) == 20
    assert len(snapshot.relationships) == 20
    assert snapshot.quality == {
        "source_team_pages": 2,
        "source_staff_records": 20,
        "canonical_coaches": 20,
        "quarantined_records": 0,
        "unsafe_source_mapping_collisions": 0,
    }
    joe_brady = next(
        entity for entity in snapshot.entities if entity["canonical_name"] == "Joe Brady"
    )
    assert joe_brady["entity_type"] == "person"
    assert joe_brady["roles"][0]["role"] == "coach"
    assert joe_brady["roles"][0]["source"] == BANK_SOURCE
    assert joe_brady["source_mappings"][0]["source_entity_id"] == "joe brady"
    assert joe_brady["aliases"] == [
        {
            "alias": "Joe Brady",
            "normalized_alias": "joe brady",
            "alias_type": "canonical_name",
        }
    ]
    relationship = next(
        item
        for item in snapshot.relationships
        if item["subject_entity_id"] == joe_brady["entity_id"]
    )
    assert relationship["predicate"] == "coaches_for"
    assert relationship["evidence"]["team_name"] == "Buffalo Bills"


def test_coach_snapshot_quarantines_cross_team_name_collisions() -> None:
    snapshot = normalize_coach_snapshot(
        (
            coach_asset("BUF", staff("Buffalo", head_coach="Same Coach")),
            coach_asset("CIN", staff("Cincinnati", head_coach="Same Coach")),
        ),
        season=2026,
        observed_at=NOW,
    )

    assert snapshot.quality["unsafe_source_mapping_collisions"] == 1
    assert all(entity["canonical_name"] != "Same Coach" for entity in snapshot.entities)
    assert snapshot.quarantined_records[0]["reason_codes"] == [
        "normalized_name_listed_by_multiple_teams"
    ]


def test_coach_snapshot_rejects_incomplete_official_page() -> None:
    with pytest.raises(ValueError, match="completeness checks"):
        normalize_coach_snapshot(
            (coach_asset("BUF", [("Joe Brady", "Head Coach")]),),
            season=2026,
            observed_at=NOW,
        )


def test_coach_sync_requires_explicit_confirmation_before_live_writes() -> None:
    assert coach_sync_main(["--apply"]) == 2
