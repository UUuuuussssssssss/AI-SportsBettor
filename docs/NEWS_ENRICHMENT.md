# News enrichment contract

News enrichment is downstream of X ingestion. It does not change the meaning or
write path of `raw_ingest_objects`, `news_events`, `news_media`,
`news_event_relationships`, or `ingest_cursors`.

## Output dimensions

Each `(news_id, enrichment_version)` result has:

- one or more unordered, queryable topic tags, with every materially applicable
  tag included;
- independent information status (`official`, `reported`, `rumor`, `opinion`,
  or `unknown`);
- independent usefulness (`high`, `medium`, `low`, or `irrelevant`);
- a factual summary, classification reason, entities, and claims;
- resolution-ready entity mentions with code-owned mention roles, verbatim
  evidence, confidence, and source references;
- source references that connect tags, entities, and claims to tweet, article,
  image, or video-frame evidence;
- provider, exact model ID, prompt version, input fingerprint, source manifest,
  token usage, warnings, timestamps, and any failure.

The initial topic taxonomy is:

- `injury_availability`
- `lineup_depth_chart`
- `roster_transaction`
- `coaching_management`
- `contract`
- `discipline_legal`
- `weather_field_conditions`
- `schedule_travel`
- `game_status_result`
- `performance_statistics`
- `market_odds`
- `league_management_rule`
- `promotional_social`
- `unrelated_other`

Tags do not have primary/secondary ranking or numeric confidence percentages.
Each tag instead has an evidence-strength category: `confident`, `neutral`, or
`unconfident`. `irrelevant` is a usefulness value rather than a topic. A post
can therefore be about promotional content while also being explicitly marked
irrelevant.

## Storage

Migration `20260731_05` adds:

- `news_enrichments`: one versioned result and its audit metadata;
- `news_enrichment_tags`: normalized tags for filtering and aggregation.

Raw source rows remain untouched. Reprocessing uses a new
`enrichment_version` or deliberately replaces the same version.

Migration `20260803_11` adds the entity-extractor version metadata used to
distinguish the new mention contract from immutable legacy enrichment rows.

## Source processing

Article processing follows only URLs directly represented in the X source
entities. Requests:

- allow only public HTTP/HTTPS addresses on standard ports;
- reject credentials and private, loopback, link-local, reserved, multicast,
  and unspecified network targets;
- revalidate every redirect;
- enforce redirect, timeout, byte, content-type, and extracted-text limits;
- do not bypass authentication or paywalls.

Images are normalized to bounded JPEG inputs for Claude Vision.

Videos are downloaded to a temporary file, sampled into at most eight bounded
frames from the first five minutes, and deleted. Complete video is not added to
the raw GCS media prefix. Claude does not accept raw video or audio, so audio is
reported as `not_transcribed` unless a transcript is supplied by another
component. The schema retains provided transcripts and makes that limitation
visible rather than claiming unsupported audio analysis.

News enrichment is the only LLM extraction pass for X evidence. It extracts
mentions from the combined tweet/article/media evidence and validates textual
mention evidence before persistence. The entity-bank worker later reads these
objects and performs only canonical candidate retrieval and, when needed,
allowlisted adjudication. It never asks a second model to extract the tweet.

## Safe local dry run

Configure `ANTHROPIC_API_KEY`, `NEWS_ENRICHMENT_MODEL`, and
`NEWS_ENRICHMENT_VERSION` in the process environment or in the ignored
`src/.env` file. Never commit `src/.env`. The pinned default model is
`claude-haiku-4-5-20251001`, which retains structured text and vision support
at a lower token price than Sonnet. Output is capped at 1,536 tokens by default;
override that only after evaluation with `NEWS_ENRICHMENT_MAX_OUTPUT_TOKENS`.
The production job worker scores tweets in small batches (default pairs) using
`NEWS_ENRICHMENT_BATCH_SIZE`. Set it to `1` to restore one Claude call per tweet,
or raise it up to `10` after checking quality. Each tweet still persists its own
`news_enrichments` row; token usage from a shared call is split across those rows.
The current resolution-ready contract uses enrichment version `v3`; older rows
remain available as historical audit records. Version `v3` tightens entity
evidence binding and distinguishes NFL roster transactions from commercial
endorsements.

Fully offline control-flow test:

```powershell
python -m src.enrich_news.dry_run `
  --input tests/fixtures/enrichment_records.jsonl `
  --provider mock
```

Live Claude with local output and optional public source reads:

```powershell
$env:ANTHROPIC_API_KEY = "<secret>"
python -m src.enrich_news.dry_run `
  --input tests/fixtures/enrichment_records.jsonl `
  --provider claude `
  --allow-network
```

The dry-run module never initializes Cloud SQL or GCS.
Mock-provider results cannot be applied to PostgreSQL.

Read existing PostgreSQL candidates but keep results local:

```powershell
python -m src.enrich_news.worker --limit 20 --provider claude --allow-network
```

Database persistence requires both `--apply` and the explicit confirmation:

```powershell
python -m src.enrich_news.worker `
  --limit 20 `
  --version v3 `
  --provider claude `
  --allow-network `
  --apply `
  --confirm-live-writes APPLY_ENRICHMENTS
```

To backfill all existing tweets, repeat the applying command in batches of up
to 100 until `record_count` is zero. Version `v3` does not overwrite prior
results.

Do not use the applying form until migrations `20260731_05` and `20260803_11`
have been reviewed and applied to the intended database.
