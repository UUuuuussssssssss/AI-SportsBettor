# AI-SportsBettor

A pipeline for collecting timestamped NFL news and building auditable datasets
for sports-market research.

## 🎯 Project Goals

- Ingest NFL news from selected X accounts with proper timestamp handling
- Preserve raw source payloads in Google Cloud Storage
- Prepare trustworthy, time-aware inputs for future market analysis
- Maintain full audit trail with raw data preservation

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip or uv package manager
- PostgreSQL 16, locally or through Google Cloud SQL

### Installation

```bash
# Clone the repository
cd AI-SportsBettor

# Install dependencies
pip install -e .

# (Optional) Copy and configure environment
cp .env.example .env
# Edit .env with your configuration
```

### Running X News Ingestion

```bash
# Continuously fetch NFL news from configured X accounts
python -m src.ingest_news.X_pull
```

This will:
- Query the accounts configured in `src/config/x_config.json`
- Paginate the current X API v2 recent-search endpoint
- Recover stale or incompatible checkpoints with a bounded lookback window
- Preserve source timestamps and ingestion timestamps
- Upload versioned gzip JSON envelopes and attached media to Google Cloud Storage

See [X ingestion and storage contract](docs/X_INGESTION.md) for the checkpoint,
GCS layout, envelope schema, and PostgreSQL mapping.
See [news enrichment contract](docs/NEWS_ENRICHMENT.md) for versioned Claude
tagging, article/image/video-frame processing, and local-only dry runs.
For Cloud SQL and the eventual Compute Engine collector, follow the
[Google Cloud Console setup](docs/GCP_SETUP.md).

### Running Polymarket NFL Discovery

```bash
# Discover currently open Gamma events tagged NFL and preserve raw pages in GCS
python -m src.ingest_odds.polymarket_pull

# Preview one real cycle without writing to GCS or PostgreSQL
python -m src.ingest_odds.polymarket_pull --dry-run

# Collect current CLOB order books for discovered open-market tokens
python -m src.ingest_odds.clob_order_book_pull
```

This collector discovers events, markets, and CLOB outcome token IDs. See
[Polymarket ingestion and storage contract](docs/POLYMARKET_INGESTION.md) for
the path convention, version-history behavior, and PostgreSQL mapping.

### Auditing and Building the NFL Entity Bank

```bash
# Live nflverse + Gamma, local files only: no database or GCS reads/writes
python -m src.entity_bank.audit --provider mock --event-limit 20

# Use schema-constrained Claude output while still making no storage writes
python -m src.entity_bank.audit --provider claude --event-limit 20

# Preview the complete canonical nflverse registry as local JSONL
python -m src.entity_bank.nflverse_sync --season 2026 --limit 100

# Preview all current coaches from the 32 official NFL club directories
python -m src.entity_bank.coach_sync --season 2026
```

The bank uses internal IDs, keeps nflverse/official-club/provider IDs as mappings, records
ambiguous and unresolved mentions instead of guessing, and never treats a
prediction or social claim as factual roster membership. See
[NFL entity bank and source resolution](docs/ENTITY_BANK.md) for schema,
dry-run outputs, explicit apply confirmations, and backfill order.

### Running Real-Time Enrichment and Resolution

```bash
# Preview/seed rows created before the queue migration
python -m src.jobs.seed

# Continuously process new work with bounded concurrency
python -m src.jobs.worker \
  --concurrency 10 \
  --confirm-live-writes RUN_JOB_WORKER
```

X, enrichment, and Gamma writes enqueue downstream work transactionally.
Leases, idempotency keys, and retries make the queue safe across process or VM
restarts. See [durable enrichment and entity worker](docs/JOB_QUEUE.md).

## 📁 Project Structure

```
AI-SportsBettor/
├── src/
│   ├── ingest_news/
│   │   └── X_pull.py     # X news ingestion
│   ├── config/
│   │   └── x_config.json
│   └── common/           # Shared utilities & config
│       ├── settings.py   # Configuration management
│       └── logging_config.py
├── data/
│   ├── raw/
│   │   └── news/         # Optional local raw news data
│   └── ref/              # Reference data (teams, schedules)
├── infra/                # Docker Compose, migrations (TODO)
├── tests/                # Unit tests (TODO)
└── notebooks/            # Analysis notebooks (TODO)
```

## 🔑 Key Features (Current)

### X News Ingestion ✅

- **Configured Sources**: Collects posts from selected NFL accounts
- **Time Truth**: Preserves post and ingestion timestamps
- **Rate-Aware Polling**: Splits handles into bounded query batches and paginates
- **Audit Trail**: Stores normalized records and exact API responses in GCS
- **Restart Safety**: Persists a timestamped, query-aware PostgreSQL cursor
- **Database Ready**: Gives each post and media attachment stable upsert keys
- **Transactional Metadata**: Commits records, media, and cursor atomically

## 🚧 Roadmap

### Week 1 (MVP)
- [x] X news ingestion
- [x] PostgreSQL/Cloud SQL connection layer
- [x] Database schema (Alembic migrations)
- [ ] APScheduler automation

### Week 2+
- [x] NFL entity bank and auditable source resolution
- [x] Polymarket Gamma event discovery
- [x] Polymarket CLOB order-book ingestion
- [ ] News-to-market linking
- [ ] Weather data integration
- [ ] Player props
- [ ] Closing line value (CLV) metrics
- [ ] Basic line movement models
- [ ] Alert generation system

## 🛡️ Non-Negotiables

- **ToS-Safe**: Use authorized APIs and respect provider terms
- **UTC Everywhere**: All timestamps in UTC
- **Audit Trail**: Preserve raw source payloads
- **Human-in-the-Loop**: No automated wagering
- **Time Truth**: Preserve source and ingestion timestamps

## 📊 Data Flow (Planned)

```
X API → Versioned raw envelope + media in GCS
                         ↓
          raw_ingest_objects / news_events / news_media
                         ↓
       Multimodal enrichment + one-pass mention extraction
                         ↓
           Canonical entity candidate resolution
                         ↓
              Polymarket Integration
                         ↓
                Time-Aligned Analysis
```

## 📝 Example Usage

### Start X Collection

```bash
python -m src.ingest_news.X_pull
```

## 🤝 Contributing

This is a personal/solo MVP project. Focus areas:
1. Keep it simple and maintainable
2. Typed Python with Pydantic models
3. Small, testable modules
4. UTC timestamps everywhere
5. Preserve raw data for audit

## 📄 License

Private project for educational/research purposes only.

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It does not place bets automatically and is not financial advice. Sports betting carries risk. Never bet more than you can afford to lose.
