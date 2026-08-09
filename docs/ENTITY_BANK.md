# NFL entity bank and source resolution

## Safety model

The entity bank owns stable internal IDs. Provider IDs are source mappings, not
database primary keys. nflverse is the automated source of factual player roster
relationships. The 32 official NFL club coaching-staff directories are the
automated source of factual coach/team relationships. A Polymarket destination
market or X claim can create a mention, but cannot change who plays or coaches
for which team.

Closed code-owned taxonomies constrain entity type, person role, mention role,
market topic, contract type, identity status, resolution status, alias type,
and match method. Claude can select values but cannot invent new labels or
entity IDs.

## Data flow

1. `nflverse_sync` discovers the current GitHub release assets, fetches exact
   team, player-history, and current season-roster CSVs, and verifies provider SHA-256 digests
   when GitHub supplies them.
2. The full historical player master is used as an identity lookup, but only
   people present in the current-season roster are placed in the active
   canonical bank. A dry run writes separate current-player and complete-history
   JSONL files.
3. An applied sync archives the exact files in one gzip GCS envelope and
   transactionally upserts the canonical registry.
4. `coach_sync` fetches the 32 official NFL club coaching-staff directories,
   requires one head coach and at least ten staff records per team, and creates
   canonical people with `coach` roles and `coaches_for` relationships. Staff
   titles and profile/directory URLs remain attached as source evidence.
5. Gamma entity fields are normalized into `game_id`, `group_item_title`, and
   `group_item_threshold`. A one-time backfill can replay the latest raw Gamma
   envelope.
6. News enrichment extracts and persists X mentions once from combined
   tweet/article/media evidence. The resolution worker consumes those stored
   mentions without a second extraction call. Polymarket extraction remains in
   the entity worker.
7. The worker retrieves a small lexical candidate set and asks Claude to choose
   only from that allowlist when deterministic matching is insufficient.
8. Results and attempts are versioned. Unresolved/ambiguous mentions are
   reconsidered after a new entity-bank version only when their candidate set
   changes.

Large Polymarket events are processed in batches of at most 20 markets with
repeated event context. Input/output market IDs must match exactly. A malformed,
omitted, duplicated, or hallucinated ID fails validation and is retried one
market at a time.

## Resolution behavior

- A unique exact or normalized alias resolves deterministically.
- Fuzzy retrieval only creates an allowlist; it never finalizes identity.
- Claude may resolve to an allowlisted ID or return `ambiguous`/`unresolved`.
- A structured, non-placeholder Polymarket person absent from nflverse may
  become `provisional`.
- X alone never creates an entity.
- `Player A`, `Coach B`, `Other`, and similar options become terminal `ignored`
  mentions. The underlying market and token remain untouched.
- A later nflverse or official-coach sync merges a provisional identity only
  when its normalized name maps to exactly one canonical alias. Collisions
  remain for review.

`entity_resolution_attempts` is append-only. The review queue is queried from
current `ambiguous` and `unresolved` mentions; it is not a second mutable queue.

## PostgreSQL tables

- `entity_bank_versions`: source snapshot and raw-object lineage
- `entities`: canonical/provisional/merged identities
- `entity_aliases`: source-attributed surface forms
- `entity_source_mappings`: nflverse, official club, GSIS, ESPN, PFR, and other provider IDs
- `entity_roles`: time/source-attributed roles
- `entity_relationships`: nflverse roster and official club coaching evidence
- `entity_mentions`: Polymarket/X evidence and current resolution
- `news_entity_resolution_runs`: idempotent news handoff, including zero-mention results
- `polymarket_market_classifications`: topic, contract type, and entity fingerprint
- `entity_resolution_attempts`: candidates, decision, model, usage, and error history

Migration `20260803_11` also adds entity-relevant Gamma fields to the current
event and market tables.

## No-write audits

### Manually triggered high-accuracy sweep

The accuracy sweep is a separate, intentionally expensive audit of existing
mention resolutions. It expands candidate retrieval, runs two independent
strong-model adjudications per mention, and reports consensus confirmations,
proposed changes, pass disagreements, low-confidence results, and errors. It
never writes PostgreSQL; proposed changes must be accepted through the
dashboard's audited manual-adjudication control.

Run a deterministic, no-network smoke test:

```bash
python -m src.entity_bank.accuracy_sweep \
  --provider mock \
  --scope needs_review \
  --limit 20
```

Run the billable strong-model sweep only after selecting a bounded scope and
limit:

```bash
python -m src.entity_bank.accuracy_sweep \
  --provider claude \
  --model claude-opus-5 \
  --scope needs_review \
  --limit 50 \
  --confirm-network RUN_ENTITY_ACCURACY_SWEEP
```

Each run writes `progress.json`, `summary.json`, and `findings.jsonl` under
`data/local/entity_accuracy_sweep/`. A hosted Cloud Run Job instead supplies
`--output-gcs-uri gs://BUCKET/entity-accuracy-sweeps/RUN_ID` so progress and
findings survive container shutdown and dashboard restarts. Manual review
locks are excluded. No `--apply` option exists by design.

The most complete pre-database audit fetches live nflverse and Gamma data,
optionally calls Claude, and writes only local files:

```bash
# Offline semantic fixture; validates control flow and output structure.
python -m src.entity_bank.audit \
  --provider mock \
  --event-limit 20

# Real schema-constrained model output; still no GCS or database reads/writes.
python -m src.entity_bank.audit \
  --provider claude \
  --event-limit 20
```

Each run writes `summary.json`, `classifications.jsonl`, `mentions.jsonl`,
`provisional_entities.jsonl`, `resolution_attempts.jsonl`,
`news_resolution_runs.jsonl`, and `failures.jsonl` under
`data/local/entity_bank/`.

To audit only the canonical source:

```bash
python -m src.entity_bank.nflverse_sync --season 2026 --limit 100

# Preview all current coaching staffs from the 32 official club directories.
python -m src.entity_bank.coach_sync --season 2026
```

This performs network reads but no GCS or database reads/writes.
`current_players.jsonl` contains only current-season rostered people;
`proposed_entities.jsonl` contains those people plus current teams for the
database; and `complete_player_history.jsonl` is the one local bookkeeping copy
of the normalized historical player master. The exact source files are also
preserved in an applied GCS envelope, while PostgreSQL receives only proposed
current entities. The summary reports source-row coverage, roster-only people,
unmatched roster rows, and normalized alias collisions. It also writes
`quarantined_records.jsonl` and `source_mapping_conflicts.jsonl`. Compatible
player rows sharing an identifier, name, and non-conflicting birth date are
merged. Contradictory roster rows are excluded from both entities and
relationships rather than guessed.

The coach dry run writes `current_coaches.jsonl`,
`proposed_relationships.jsonl`, `quarantined_records.jsonl`, and `summary.json`
under `data/local/entity_bank/coaches_*`. The official directory name is stored
as a canonical alias. Alternate punctuation that normalizes identically is
already handled by the shared name normalizer; unsupported nicknames are not
invented and can be added later as source-attributed or manual aliases. A name
listed by multiple teams is quarantined rather than merged by name alone.

After migration, preview the Gamma replay without changing PostgreSQL:

```bash
python -m src.entity_bank.gamma_backfill
```

After a canonical bank exists, preview database-backed incremental work:

```bash
python -m src.entity_bank.worker \
  --source both \
  --provider claude \
  --event-limit 20 \
  --news-limit 20
```

This reads PostgreSQL and calls the model only for ambiguous candidate
adjudication; X mention extraction has already occurred in news enrichment.
It writes only local audit files. No entity command writes GCS or PostgreSQL
merely because it was started.

## Apply sequence

Review the local JSONL first, then apply explicit, separately confirmed steps:

```bash
alembic upgrade head

python -m src.entity_bank.nflverse_sync \
  --season 2026 \
  --apply \
  --confirm-live-writes APPLY_NFLVERSE_ENTITY_BANK

python -m src.entity_bank.coach_sync \
  --season 2026 \
  --apply \
  --confirm-live-writes APPLY_NFL_COACH_ENTITY_BANK

python -m src.entity_bank.gamma_backfill \
  --apply \
  --confirm-live-writes APPLY_GAMMA_ENTITY_BACKFILL

# Repeat in bounded batches until record_count is zero.
python -m src.enrich_news.worker \
  --version v3 \
  --limit 100 \
  --provider claude \
  --allow-network \
  --apply \
  --confirm-live-writes APPLY_ENRICHMENTS

python -m src.entity_bank.worker \
  --source both \
  --provider claude \
  --apply \
  --confirm-live-writes APPLY_ENTITY_RESOLUTIONS
```

The mock provider is rejected with `--apply`. Both canonical-source syncs reject
`--limit` with `--apply`, so a partial bank cannot be written accidentally. The
coach sync also requires all canonical team identities to exist first. Any
remaining source identifier collision aborts before coach rows are written.

## Continuous nflverse polling

The production poller checks nflverse once per day by default. It downloads and
hashes the source assets, but creates no GCS object and performs no PostgreSQL
writes when the structural snapshot hash is unchanged. A changed snapshot is
archived to GCS first and then transactionally upserted into PostgreSQL. Poll
cycles are sequential, so a slow cycle can never overlap the next cycle.

Smoke-test one live cycle before installing the service:

```bash
python -m src.entity_bank.nflverse_poll \
  --once \
  --confirm-live-writes RUN_NFLVERSE_ENTITY_POLL
```

The season is inferred automatically: January and February remain on the prior
NFL season, and March starts the new league year. `--season` is available as an
explicit override. The minimum interval is five minutes, but daily polling is
recommended because nflverse roster releases are not real-time.

Example `/etc/systemd/system/entity-bank-nflverse.service`:

```ini
[Unit]
Description=AI Sports Bettor nflverse Entity Bank Poller
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=sportsbettor
Group=sportsbettor
WorkingDirectory=/opt/ai-sports-bettor
EnvironmentFile=/etc/ai-sports-bettor/x.env
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/ai-sports-bettor/.venv/bin/python -m src.entity_bank.nflverse_poll --confirm-live-writes RUN_NFLVERSE_ENTITY_POLL
Restart=on-failure
RestartSec=30
KillSignal=SIGINT
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

The poller writes one structured `NFLVERSE_POLL_RESULT` log per successful
cycle. It does not retain full local JSONL copies on the VM; changed source
snapshots are already retained in GCS.

PostgreSQL stores current-season rostered players and teams, not the complete
historical player master. Storage grows by distinct active entities, aliases,
source IDs, roles, and roster relationships—not by poll count. Repeated polls
upsert existing rows. An unchanged poll adds zero rows. A changed roster
generally adds at most one relationship per player/team/season, roughly
thousands per season rather than millions per day; PostgreSQL can comfortably
handle this scale with the existing uniqueness constraints and indexes.

## Known source limitations

nflverse's player master and current roster may not update atomically. Roster
people missing from the player master are retained as canonical roster-only
people when a stable source ID is available. Rows with no usable person ID are
reported and are not guessed. Rows whose identifiers resolve to different
people are quarantined with the matching entities and source evidence.
Name suffix and common source-name variants are retained as aliases when the
identifier owner is otherwise consistent. Duplicate names and initials remain
separate internal identities and ambiguous aliases never auto-merge.

Confirm nflverse and underlying provider licensing/retention terms before
production use.

Official club pages provide current directory names, titles, and sometimes
profile URLs, but do not provide a universal immutable coach ID or a complete
nickname/pronunciation registry. Coach identities therefore use a normalized
official name within a collision-checked snapshot. A cross-team name collision
blocks the applied sync and must be adjudicated instead of guessed. Exact source
HTML is archived in GCS for auditability.
