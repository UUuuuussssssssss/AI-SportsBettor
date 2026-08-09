"""Transactional persistence and candidate reads for the entity bank."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from src.db.engine import DatabaseResources, create_database_resources
from src.db.models import (
    entities,
    entity_aliases,
    entity_bank_versions,
    entity_mentions,
    entity_relationships,
    entity_roles,
    entity_source_mappings,
    raw_ingest_objects,
)
from src.db.repository import raw_object_values
from src.entity_bank.coach_pipeline import BANK_SOURCE as COACH_BANK_SOURCE
from src.entity_bank.models import IdentityStatus
from src.entity_bank.nflverse_pipeline import ENTITY_NAMESPACE


def _chunks(values: list[dict[str, Any]], size: int = 1000) -> Iterable[list[dict[str, Any]]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


class EntityBankRepository:
    def __init__(self, resources: DatabaseResources) -> None:
        self.resources = resources

    @classmethod
    def from_environment(cls, src_dir: Path) -> EntityBankRepository:
        return cls(create_database_resources(src_dir))

    def close(self) -> None:
        self.resources.close()

    def latest_content_sha256(self, source: str = "nflverse") -> str | None:
        with self.resources.engine.connect() as connection:
            return connection.scalar(
                select(entity_bank_versions.c.content_sha256)
                .where(
                    entity_bank_versions.c.source == source,
                    entity_bank_versions.c.status == "completed",
                )
                .order_by(entity_bank_versions.c.ingested_at.desc())
                .limit(1)
            )

    def persist_nflverse_snapshot(self, envelope: dict[str, Any]) -> dict[str, int | str]:
        observed_at = datetime.fromisoformat(envelope["ingested_at"]).astimezone(UTC)
        version_id = envelope["content_sha256"][:32]
        normalized_entities = envelope["_normalized_entities"]
        normalized_relationships = envelope["_normalized_relationships"]

        entity_rows: list[dict[str, Any]] = []
        alias_rows: list[dict[str, Any]] = []
        mapping_rows: list[dict[str, Any]] = []
        role_rows: list[dict[str, Any]] = []
        for entity in normalized_entities:
            entity_rows.append(
                {
                    "entity_id": entity["entity_id"],
                    "entity_type": entity["entity_type"],
                    "canonical_name": entity["canonical_name"],
                    "normalized_name": entity["normalized_name"],
                    "identity_status": IdentityStatus.CANONICAL.value,
                    "merged_into_entity_id": None,
                    "latest_bank_version_id": version_id,
                    "first_observed_at": observed_at,
                    "last_observed_at": observed_at,
                    "updated_at": observed_at,
                }
            )
            for alias in entity["aliases"]:
                alias_rows.append(
                    {
                        "entity_id": entity["entity_id"],
                        "normalized_alias": alias["normalized_alias"],
                        "source": "nflverse",
                        "alias": alias["alias"],
                        "alias_type": alias["alias_type"],
                        "confidence": 1,
                        "first_observed_at": observed_at,
                        "last_observed_at": observed_at,
                        "source_metadata": {},
                    }
                )
            for mapping in entity["source_mappings"]:
                mapping_rows.append(
                    {
                        "provider": mapping["provider"],
                        "source_entity_type": mapping["source_entity_type"],
                        "source_entity_id": mapping["source_entity_id"],
                        "entity_id": entity["entity_id"],
                        "first_observed_at": observed_at,
                        "last_observed_at": observed_at,
                        "source_metadata": mapping["metadata"],
                    }
                )
            for role in entity["roles"]:
                role_key = (
                    f"role:{entity['entity_id']}:{role['role']}:{role['source']}:"
                    f"{envelope['request']['season']}"
                )
                role_rows.append(
                    {
                        "role_id": str(uuid.uuid5(ENTITY_NAMESPACE, role_key)),
                        "entity_id": entity["entity_id"],
                        "role": role["role"],
                        "source": role["source"],
                        "valid_from": None,
                        "valid_to": None,
                        "confidence": 1,
                        "evidence": role["evidence"],
                    }
                )
        relationship_rows = [
            {
                **relationship,
                "valid_from": observed_at,
                "valid_to": None,
                "confidence": 1,
            }
            for relationship in normalized_relationships
        ]

        reconciled_provisionals = 0
        with self.resources.engine.begin() as connection:
            connection.execute(
                insert(raw_ingest_objects)
                .values(**raw_object_values(envelope))
                .on_conflict_do_nothing(index_elements=[raw_ingest_objects.c.ingest_run_id])
            )
            connection.execute(
                insert(entity_bank_versions)
                .values(
                    version_id=version_id,
                    raw_ingest_run_id=envelope["ingest_run_id"],
                    source="nflverse",
                    season=envelope["request"]["season"],
                    content_sha256=envelope["content_sha256"],
                    status="completed",
                    source_metadata={"assets": envelope["request"]["assets"]},
                    ingested_at=observed_at,
                )
                .on_conflict_do_nothing(index_elements=[entity_bank_versions.c.version_id])
            )
            current_relationship_keys = {
                row["source_key"] for row in relationship_rows
            }
            stale_relationships = (
                update(entity_relationships)
                .where(
                    entity_relationships.c.source == "nflverse",
                    entity_relationships.c.predicate == "rostered_by",
                    entity_relationships.c.valid_to.is_(None),
                    entity_relationships.c.source_key.like(
                        f"nflverse:roster:{envelope['request']['season']}:%"
                    ),
                )
                .values(valid_to=observed_at)
            )
            if current_relationship_keys:
                stale_relationships = stale_relationships.where(
                    entity_relationships.c.source_key.not_in(
                        current_relationship_keys
                    )
                )
            connection.execute(stale_relationships)
            entity_statement = insert(entities)
            entity_upsert = entity_statement.on_conflict_do_update(
                index_elements=[entities.c.entity_id],
                set_={
                    "entity_type": entity_statement.excluded.entity_type,
                    "canonical_name": entity_statement.excluded.canonical_name,
                    "normalized_name": entity_statement.excluded.normalized_name,
                    "latest_bank_version_id": entity_statement.excluded.latest_bank_version_id,
                    "last_observed_at": entity_statement.excluded.last_observed_at,
                    "updated_at": entity_statement.excluded.updated_at,
                },
                where=entities.c.identity_status == IdentityStatus.CANONICAL.value,
            )
            for chunk in _chunks(entity_rows):
                connection.execute(entity_upsert.values(chunk))

            alias_statement = insert(entity_aliases)
            alias_upsert = alias_statement.on_conflict_do_update(
                index_elements=[
                    entity_aliases.c.entity_id,
                    entity_aliases.c.normalized_alias,
                    entity_aliases.c.source,
                ],
                set_={
                    "alias": alias_statement.excluded.alias,
                    "alias_type": alias_statement.excluded.alias_type,
                    "confidence": alias_statement.excluded.confidence,
                    "last_observed_at": alias_statement.excluded.last_observed_at,
                    "source_metadata": alias_statement.excluded.source_metadata,
                },
            )
            for chunk in _chunks(alias_rows):
                connection.execute(alias_upsert.values(chunk))

            mapping_statement = insert(entity_source_mappings)
            mapping_upsert = mapping_statement.on_conflict_do_update(
                index_elements=[
                    entity_source_mappings.c.provider,
                    entity_source_mappings.c.source_entity_type,
                    entity_source_mappings.c.source_entity_id,
                ],
                set_={
                    "last_observed_at": mapping_statement.excluded.last_observed_at,
                    "source_metadata": mapping_statement.excluded.source_metadata,
                },
            )
            for chunk in _chunks(mapping_rows):
                connection.execute(mapping_upsert.values(chunk))

            role_statement = insert(entity_roles)
            role_upsert = role_statement.on_conflict_do_update(
                index_elements=[entity_roles.c.role_id],
                set_={
                    "confidence": role_statement.excluded.confidence,
                    "evidence": role_statement.excluded.evidence,
                },
            )
            for chunk in _chunks(role_rows):
                connection.execute(role_upsert.values(chunk))

            relationship_statement = insert(entity_relationships)
            relationship_upsert = relationship_statement.on_conflict_do_update(
                index_elements=[entity_relationships.c.source_key],
                set_={
                    "evidence": relationship_statement.excluded.evidence,
                    "confidence": relationship_statement.excluded.confidence,
                    "valid_to": relationship_statement.excluded.valid_to,
                },
            )
            for chunk in _chunks(relationship_rows):
                connection.execute(relationship_upsert.values(chunk))

            canonical_alias_rows = connection.execute(
                select(
                    entity_aliases.c.normalized_alias,
                    entity_aliases.c.entity_id,
                )
                .join(entities, entities.c.entity_id == entity_aliases.c.entity_id)
                .where(
                    entities.c.identity_status == IdentityStatus.CANONICAL.value,
                    entities.c.latest_bank_version_id == version_id,
                )
            ).all()
            canonical_by_alias: dict[str, set[str]] = {}
            for normalized_alias, entity_id in canonical_alias_rows:
                canonical_by_alias.setdefault(normalized_alias, set()).add(entity_id)
            provisional_rows = connection.execute(
                select(entities.c.entity_id, entities.c.normalized_name).where(
                    entities.c.identity_status == IdentityStatus.PROVISIONAL.value
                )
            ).all()
            for provisional_id, normalized_name in provisional_rows:
                canonical_ids = canonical_by_alias.get(normalized_name, set())
                if len(canonical_ids) != 1:
                    continue
                canonical_id = next(iter(canonical_ids))
                connection.execute(
                    update(entities)
                    .where(
                        entities.c.entity_id == provisional_id,
                        entities.c.identity_status == IdentityStatus.PROVISIONAL.value,
                    )
                    .values(
                        identity_status=IdentityStatus.MERGED.value,
                        merged_into_entity_id=canonical_id,
                        updated_at=observed_at,
                    )
                )
                connection.execute(
                    update(entity_mentions)
                    .where(entity_mentions.c.entity_id == provisional_id)
                    .values(
                        entity_id=canonical_id,
                        match_method="normalized_alias",
                        last_bank_version_id=version_id,
                        resolution_metadata={
                            "reason": "Provisional identity reconciled by a unique "
                            "canonical nflverse alias.",
                            "merged_from_entity_id": provisional_id,
                        },
                        updated_at=observed_at,
                    )
                )
                reconciled_provisionals += 1

        return {
            "version_id": version_id,
            "entities": len(entity_rows),
            "aliases": len(alias_rows),
            "mappings": len(mapping_rows),
            "roles": len(role_rows),
            "relationships": len(relationship_rows),
            "reconciled_provisionals": reconciled_provisionals,
        }

    def persist_coach_snapshot(self, envelope: dict[str, Any]) -> dict[str, int | str]:
        """Persist one complete official-club coaching-staff snapshot."""

        observed_at = datetime.fromisoformat(envelope["ingested_at"]).astimezone(UTC)
        version_id = envelope["content_sha256"][:32]
        season = int(envelope["request"]["season"])
        normalized_entities = envelope["_normalized_entities"]
        normalized_relationships = envelope["_normalized_relationships"]

        entity_rows: list[dict[str, Any]] = []
        alias_rows: list[dict[str, Any]] = []
        mapping_rows: list[dict[str, Any]] = []
        role_rows: list[dict[str, Any]] = []
        for entity in normalized_entities:
            entity_rows.append(
                {
                    "entity_id": entity["entity_id"],
                    "entity_type": entity["entity_type"],
                    "canonical_name": entity["canonical_name"],
                    "normalized_name": entity["normalized_name"],
                    "identity_status": IdentityStatus.CANONICAL.value,
                    "merged_into_entity_id": None,
                    "latest_bank_version_id": version_id,
                    "first_observed_at": observed_at,
                    "last_observed_at": observed_at,
                    "updated_at": observed_at,
                }
            )
            for alias in entity["aliases"]:
                alias_rows.append(
                    {
                        "entity_id": entity["entity_id"],
                        "normalized_alias": alias["normalized_alias"],
                        "source": COACH_BANK_SOURCE,
                        "alias": alias["alias"],
                        "alias_type": alias["alias_type"],
                        "confidence": 1,
                        "first_observed_at": observed_at,
                        "last_observed_at": observed_at,
                        "source_metadata": {},
                    }
                )
            for mapping in entity["source_mappings"]:
                mapping_rows.append(
                    {
                        "provider": mapping["provider"],
                        "source_entity_type": mapping["source_entity_type"],
                        "source_entity_id": mapping["source_entity_id"],
                        "entity_id": entity["entity_id"],
                        "first_observed_at": observed_at,
                        "last_observed_at": observed_at,
                        "source_metadata": mapping["metadata"],
                    }
                )
            for role in entity["roles"]:
                team_entity_id = next(
                    relationship["object_entity_id"]
                    for relationship in normalized_relationships
                    if relationship["subject_entity_id"] == entity["entity_id"]
                )
                role_key = (
                    f"role:{entity['entity_id']}:{role['role']}:{role['source']}:"
                    f"{season}:{team_entity_id}"
                )
                role_rows.append(
                    {
                        "role_id": str(uuid.uuid5(ENTITY_NAMESPACE, role_key)),
                        "entity_id": entity["entity_id"],
                        "role": role["role"],
                        "source": role["source"],
                        "valid_from": observed_at,
                        "valid_to": None,
                        "confidence": 1,
                        "evidence": role["evidence"],
                    }
                )
        relationship_rows = [
            {
                **relationship,
                "valid_from": observed_at,
                "valid_to": None,
                "confidence": 1,
            }
            for relationship in normalized_relationships
        ]

        reconciled_provisionals = 0
        with self.resources.engine.begin() as connection:
            required_team_ids = {
                relationship["object_entity_id"] for relationship in relationship_rows
            }
            existing_team_ids = set(
                connection.scalars(
                    select(entities.c.entity_id).where(
                        entities.c.entity_id.in_(required_team_ids),
                        entities.c.entity_type == "team",
                        entities.c.identity_status == IdentityStatus.CANONICAL.value,
                    )
                )
            )
            missing_team_ids = sorted(required_team_ids - existing_team_ids)
            if missing_team_ids:
                raise RuntimeError(
                    "coach sync requires the canonical nflverse teams first; "
                    f"missing {len(missing_team_ids)} team identities"
                )

            connection.execute(
                insert(raw_ingest_objects)
                .values(**raw_object_values(envelope))
                .on_conflict_do_nothing(index_elements=[raw_ingest_objects.c.ingest_run_id])
            )
            connection.execute(
                insert(entity_bank_versions)
                .values(
                    version_id=version_id,
                    raw_ingest_run_id=envelope["ingest_run_id"],
                    source=COACH_BANK_SOURCE,
                    season=season,
                    content_sha256=envelope["content_sha256"],
                    status="completed",
                    source_metadata={"assets": envelope["request"]["assets"]},
                    ingested_at=observed_at,
                )
                .on_conflict_do_nothing(index_elements=[entity_bank_versions.c.version_id])
            )

            current_relationship_keys = {row["source_key"] for row in relationship_rows}
            stale_relationships = (
                update(entity_relationships)
                .where(
                    entity_relationships.c.source == COACH_BANK_SOURCE,
                    entity_relationships.c.predicate == "coaches_for",
                    entity_relationships.c.valid_to.is_(None),
                )
                .values(valid_to=observed_at)
            )
            if current_relationship_keys:
                stale_relationships = stale_relationships.where(
                    entity_relationships.c.source_key.not_in(current_relationship_keys)
                )
            connection.execute(stale_relationships)

            current_role_ids = {row["role_id"] for row in role_rows}
            stale_roles = (
                update(entity_roles)
                .where(
                    entity_roles.c.source == COACH_BANK_SOURCE,
                    entity_roles.c.role == "coach",
                    entity_roles.c.valid_to.is_(None),
                )
                .values(valid_to=observed_at)
            )
            if current_role_ids:
                stale_roles = stale_roles.where(entity_roles.c.role_id.not_in(current_role_ids))
            connection.execute(stale_roles)

            entity_statement = insert(entities)
            entity_upsert = entity_statement.on_conflict_do_update(
                index_elements=[entities.c.entity_id],
                set_={
                    "entity_type": entity_statement.excluded.entity_type,
                    "canonical_name": entity_statement.excluded.canonical_name,
                    "normalized_name": entity_statement.excluded.normalized_name,
                    "latest_bank_version_id": entity_statement.excluded.latest_bank_version_id,
                    "last_observed_at": entity_statement.excluded.last_observed_at,
                    "updated_at": entity_statement.excluded.updated_at,
                },
                where=entities.c.identity_status == IdentityStatus.CANONICAL.value,
            )
            for chunk in _chunks(entity_rows):
                connection.execute(entity_upsert.values(chunk))

            alias_statement = insert(entity_aliases)
            alias_upsert = alias_statement.on_conflict_do_update(
                index_elements=[
                    entity_aliases.c.entity_id,
                    entity_aliases.c.normalized_alias,
                    entity_aliases.c.source,
                ],
                set_={
                    "alias": alias_statement.excluded.alias,
                    "alias_type": alias_statement.excluded.alias_type,
                    "confidence": alias_statement.excluded.confidence,
                    "last_observed_at": alias_statement.excluded.last_observed_at,
                    "source_metadata": alias_statement.excluded.source_metadata,
                },
            )
            for chunk in _chunks(alias_rows):
                connection.execute(alias_upsert.values(chunk))

            mapping_statement = insert(entity_source_mappings)
            mapping_upsert = mapping_statement.on_conflict_do_update(
                index_elements=[
                    entity_source_mappings.c.provider,
                    entity_source_mappings.c.source_entity_type,
                    entity_source_mappings.c.source_entity_id,
                ],
                set_={
                    "last_observed_at": mapping_statement.excluded.last_observed_at,
                    "source_metadata": mapping_statement.excluded.source_metadata,
                },
            )
            for chunk in _chunks(mapping_rows):
                connection.execute(mapping_upsert.values(chunk))

            role_statement = insert(entity_roles)
            role_upsert = role_statement.on_conflict_do_update(
                index_elements=[entity_roles.c.role_id],
                set_={
                    "valid_to": None,
                    "confidence": role_statement.excluded.confidence,
                    "evidence": role_statement.excluded.evidence,
                },
            )
            for chunk in _chunks(role_rows):
                connection.execute(role_upsert.values(chunk))

            relationship_statement = insert(entity_relationships)
            relationship_upsert = relationship_statement.on_conflict_do_update(
                index_elements=[entity_relationships.c.source_key],
                set_={
                    "evidence": relationship_statement.excluded.evidence,
                    "confidence": relationship_statement.excluded.confidence,
                    "valid_to": None,
                },
            )
            for chunk in _chunks(relationship_rows):
                connection.execute(relationship_upsert.values(chunk))

            canonical_alias_rows = connection.execute(
                select(
                    entity_aliases.c.normalized_alias,
                    entity_aliases.c.entity_id,
                )
                .join(entities, entities.c.entity_id == entity_aliases.c.entity_id)
                .where(
                    entities.c.identity_status == IdentityStatus.CANONICAL.value,
                    entities.c.latest_bank_version_id == version_id,
                )
            ).all()
            canonical_by_alias: dict[str, set[str]] = {}
            for normalized_alias, entity_id in canonical_alias_rows:
                canonical_by_alias.setdefault(normalized_alias, set()).add(entity_id)
            provisional_rows = connection.execute(
                select(entities.c.entity_id, entities.c.normalized_name).where(
                    entities.c.identity_status == IdentityStatus.PROVISIONAL.value
                )
            ).all()
            for provisional_id, normalized_name in provisional_rows:
                canonical_ids = canonical_by_alias.get(normalized_name, set())
                if len(canonical_ids) != 1:
                    continue
                canonical_id = next(iter(canonical_ids))
                connection.execute(
                    update(entities)
                    .where(
                        entities.c.entity_id == provisional_id,
                        entities.c.identity_status == IdentityStatus.PROVISIONAL.value,
                    )
                    .values(
                        identity_status=IdentityStatus.MERGED.value,
                        merged_into_entity_id=canonical_id,
                        updated_at=observed_at,
                    )
                )
                connection.execute(
                    update(entity_mentions)
                    .where(entity_mentions.c.entity_id == provisional_id)
                    .values(
                        entity_id=canonical_id,
                        match_method="normalized_alias",
                        last_bank_version_id=version_id,
                        resolution_metadata={
                            "reason": "Provisional identity reconciled by a unique "
                            "canonical official-club coach alias.",
                            "merged_from_entity_id": provisional_id,
                        },
                        updated_at=observed_at,
                    )
                )
                reconciled_provisionals += 1

        return {
            "version_id": version_id,
            "entities": len(entity_rows),
            "aliases": len(alias_rows),
            "mappings": len(mapping_rows),
            "roles": len(role_rows),
            "relationships": len(relationship_rows),
            "reconciled_provisionals": reconciled_provisionals,
        }
