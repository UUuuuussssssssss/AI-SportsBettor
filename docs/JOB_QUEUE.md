# Durable enrichment and entity worker

The production worker consumes PostgreSQL-backed jobs created in the same
transaction as the data that caused them:

- X finalization enqueues `enrich_news`.
- Completed enrichment enqueues `resolve_news`.
- A structurally changed Gamma event enqueues `resolve_market`.

`job_outbox` uses idempotency keys, atomic `FOR UPDATE SKIP LOCKED` claims,
leases, exponential retry delays, and a terminal `dead` state after five
attempts. PostgreSQL notifications are emitted after commit for optional
listeners. The bundled worker uses a one-second durable-table poll, so missed
notifications cannot lose work and processing latency remains bounded.

## Concurrency

The default is 10 worker threads. Each thread owns its Anthropic clients.
`enrich_news` jobs are grouped by `NEWS_ENRICHMENT_BATCH_SIZE` (default 2) so
one Haiku call can score a pair. Additional semaphores cap video jobs at 2 and
Polymarket event jobs at 5, which keeps CPU, memory, network, and Cloud SQL
pressure bounded on an `e2-small`. The hard CLI maximum is 30, but increasing
it is not recommended without observing rate limits and VM memory.

## First deployment

Apply both the entity-bank and queue migrations:

```bash
alembic upgrade head
```

Preview and then seed work for rows created before transactional enqueueing:

```bash
python -m src.jobs.seed
python -m src.jobs.seed \
  --limit 5 \
  --apply \
  --confirm-live-writes SEED_JOB_QUEUE
```

Use `--limit 5` for the first live smoke test. After auditing those results,
run the applying command again without `--limit` to enqueue the remaining
backlog.

Run the worker:

```bash
python -m src.jobs.worker \
  --concurrency 10 \
  --video-concurrency 2 \
  --market-concurrency 5 \
  --confirm-live-writes RUN_JOB_WORKER
```

`--once` drains the current queue and exits. The normal service mode stays
running and claims new jobs within one second.

## systemd

Use a separate unit from the collectors:

```ini
[Unit]
Description=AI Sports Bettor enrichment and entity worker
After=network-online.target

[Service]
Type=simple
User=alex
WorkingDirectory=/opt/ai-sports-bettor
EnvironmentFile=/etc/ai-sports-bettor/x.env
ExecStart=/opt/ai-sports-bettor/.venv/bin/python -m src.jobs.worker --concurrency 10 --video-concurrency 2 --market-concurrency 5 --confirm-live-writes RUN_JOB_WORKER
Restart=always
RestartSec=10
TimeoutStopSec=930

[Install]
WantedBy=multi-user.target
```

On `SIGTERM`, the worker stops claiming and waits for in-flight jobs. If the VM
dies, expired leases become claimable again. Enrichment and resolution writes
are idempotent, so replay after a crash is safe.

Useful inspection query:

```sql
SELECT job_type, status, count(*), min(available_at)
FROM job_outbox
GROUP BY job_type, status
ORDER BY job_type, status;
```
