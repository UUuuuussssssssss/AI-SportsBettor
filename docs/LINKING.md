# Tweet-Market Linking and Label Construction

This layer turns the collected raw data into training examples. It joins
tweets to the Polymarket and Kalshi markets they may affect, measures how
prices moved after each tweet, and exports the result as a flat dataset a
training run can consume directly.

Both platforms flow through the same pipeline: `news_market_links` and
`news_market_reactions` carry a `platform` column (`polymarket` /
`kalshi`), `market_id` holds the Gamma id or the Kalshi ticker, and for
Kalshi `token_id` is the market ticker itself (one yes-side book per
market, `outcome_index` 0).

Everything here is derived and rebuildable: the inputs (tweets, mentions,
order-book envelopes, trades, resolutions) are durably collected by the
ingestion services, so links and labels can be recomputed from scratch at
any time with no data loss.

## Flow

```
entity_mentions (news side)  ─┐
                              ├─> linker ──> news_market_links
entity_mentions (market side) ┘                    │
                                                   v
raw_ingest_objects + GCS order-book envelopes ─> reaction builder ──> news_market_reactions
polymarket_trades ────────────────────────────┘                            │
                                                                           v
news_events + news_enrichments + polymarket_markets ────────────> dataset export (parquet/JSONL)
```

## 1. Linker (`python -m src.linking.link_pull`)

Joins resolved news mentions to resolved market mentions on canonical
`entity_id`. The join is deterministic and mechanical — no LLM calls. A
tweet is linked only to markets that were already observed and still
tradable at publish time. Markets listed after the tweet are not linked;
new listings do not walk back through old tweets.

Per `(news_id, market_id)` pair, `news_market_links` stores:

- `shared_entity_ids` / `shared_entity_count` — which canonical entities the
  tweet and market have in common.
- `news_mention_roles` / `market_mention_roles` — per-entity mention roles
  and person-role hints from each side.
- `market_topic` / `contract_type` — from the market classification when one
  exists.
- `market_open_at_publish` — always true for rows this version writes.
  Existing false rows are deleted on the next live run.
- `platform` — which exchange the market belongs to.
- `linker_version` — currently `entity_overlap_v3`.

Exclusions: markets first observed after the tweet, and markets that were
provably final before the tweet published. For Polymarket that means
`closed_time` or `resolution_observed_at` earlier than `published_at`; for
Kalshi, `close_time` or `settlement_ts`. `first_observed_at` is our ingest
clock, not the venue listing time.

Hourly runs are incremental: only tweets that are not yet in
`news_market_links` (with a 24-hour lookback plus newly completed
resolutions) are joined to the current book. `--full` rebuilds every tweet
and prunes rows that were not regenerated. The run checkpoint lives in
`ingest_cursors` under `(linking, news_market_links)`.

`--dry-run` computes and reports links without writing.

## 2. Reaction builder (`python -m src.linking.reactions_pull`)

Builds price-reaction labels for links whose post-publish window has fully
elapsed and which have no rows at the current `label_version`
(`midpoint_reaction_v2`).

For each linked tweet it selects order-book envelopes from the
`raw_ingest_objects` index:

- **Baseline** — the newest snapshot at or before publish, within a
  10-minute lookback.
- **Horizons** — the oldest snapshot at or after publish +1m, +5m, +30m,
  and +2h, accepted only within a 120-second tolerance so a collector gap
  cannot masquerade as a later price.

Only those chosen envelopes are downloaded from GCS (at most five per
tweet, LRU-cached across the run), never the full window. Midpoints are
extracted per outcome token and stored with deltas against the baseline in
`news_market_reactions`. The baseline snapshot also contributes book
quality per token — `baseline_spread` and `baseline_bid_depth` /
`baseline_ask_depth` (executable notional per side) — because a midpoint
on a near-empty book is not a real price and labels must be able to
exclude it.

Trust signals per row instead of failures:

- `snapshot_count` / `max_gap_seconds` — coverage of the 2-hour post window
  computed from the envelope index (including window edges). A row with a
  large gap is visible, not silently wrong.
- Missing snapshots or tokens absent from an envelope leave the midpoint
  NULL rather than inventing a price.
- `trade_count` / `trade_notional` — executed volume in the 2-hour window
  from `polymarket_trades` (or `kalshi_trades`, where notional is
  `yes_price * count`). NULL when the window predates that platform's trade
  collection (Polymarket 2026-08-04, Kalshi 2026-08-05); zero means trades
  were collected and none occurred.

Envelopes are selected per platform: Polymarket links read the
`(polymarket, clob, order-books)` stream and Kalshi links the
`(kalshi, trade-api, order-books)` stream, whose records carry the same
midpoint/spread/depth fields keyed by ticker instead of token id.

`--limit N` caps the number of links labeled in one run.

## 3. Dataset export (`python -m src.linking.export_dataset`)

Joins tweet text, the latest completed enrichment (summary, claims,
usefulness), link features, reaction labels, and the market's final
resolution into one row per (tweet, market, outcome token). Writes JSONL
and parquet under `data/local/datasets/` by default.

The export is platform-aware: every row carries `platform`, and Kalshi rows
map their fields into the shared schema (`question` ← market title,
`group_item_title` ← `yes_sub_title`, `line` ← `floor_strike`,
`market_result`/`settlement_value` ← Kalshi settlement, with
`winning_outcome_index` 0 for a `yes` result so `outcome_won` works
unchanged; `uma_resolution_status` stays NULL).

- `--since` / `--until` bound the tweet publish time.
- `--format jsonl|parquet|both` (default both).
- Nested values (claims, role maps, entity lists) are JSON strings so the
  schema stays flat.
- `outcome_won` is derived from `winning_outcome_index` when the market has
  resolved; NULL otherwise.

## 4. Label buckets (`python -m src.linking.labels`)

Turns the raw deltas in an export into the classification target
(`up` / `down` / `flat`). Buckets are a formula over stored midpoints, not
data — rerunning with different knobs rebuilds every label in seconds, so
sweeping is the expected workflow.

Rule, per row:

- Book-quality gate: baseline spread must be known and at most
  `--max-spread-cents` (default 5), and each side of the baseline book
  must hold at least `--min-depth` USD (default 100) — otherwise the
  midpoint is not a real price and the row is unlabeled. Disable with
  `--no-book-quality` (required for pre-v2 exports).
- No delta at the horizon (no price coverage) → unlabeled.
- |delta| below the threshold, or inside half the baseline spread
  (quote noise by construction) → `flat`.
- Otherwise `up`/`down` by sign, **if** the window contains at least
  `--min-trades` trades (default 2) and `--min-trade-notional` USD of
  executed volume (default 50). Less than that → `flat` (a single tiny
  trade repricing a dead book is not a market reaction); unknown trades
  (window predates trade collection) → unlabeled.

Usage:

```
label-training-dataset                          # +30m, 2c, gated -> adds `target`
label-training-dataset --horizon plus_2h --threshold-cents 5
label-training-dataset --min-trade-notional 100 --min-depth 250
label-training-dataset --no-trade-confirmation --no-book-quality
label-training-dataset --sweep                  # all horizon x 1c/2c/5c combos
```

Reads the newest `tweet_market_dataset_*.parquet` in `data/local/datasets/`
unless `--input` is given, writes `<input>_labeled.parquet`, and prints the
class balance per rule — paste that into the research log
(`docs/experiments/`). Sweep columns are named
`target_<horizon>_<cents>c_<tc|raw>`, e.g. `target_plus_30m_2c_tc`.

## Versioning and rebuilds

- `linker_version` and `label_version` are stamped on every row. Changing
  linking or labeling logic means bumping the version constant; the next
  run rebuilds under the new version without destroying prior labels
  (reactions key on `label_version`; links are recomputed in place).
- To force a full label rebuild at the same version, delete the affected
  `news_market_reactions` rows and rerun the builder.
- Order of operations after a schema migration: `alembic upgrade head`,
  then linker, then reaction builder, then export.

## Running

Both builders are on-demand batches, run locally or on the VM:

```
python -m src.linking.link_pull            # or: link-news-markets
python -m src.linking.reactions_pull       # or: build-reaction-labels
python -m src.linking.export_dataset       # or: export-training-dataset
```

They need the same environment as the ingestion services (Cloud SQL access
and, for the reaction builder, GCS read access).

On the VM, `news-market-linking.timer` runs both builders hourly as a
oneshot `news-market-linking.service` (linker first, then the reaction
builder). Because both are idempotent, manual runs and the timer can
coexist safely. The export remains on-demand only.

Prefer running the reaction builder on the VM rather than locally: it
downloads order-book envelopes from GCS, and the VM sits in the same region
as the bucket and Cloud SQL, so a build that takes ~an hour over a home
connection completes in minutes there.
