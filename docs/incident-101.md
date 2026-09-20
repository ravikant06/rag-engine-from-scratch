# Incident 101 – Payment failures after payment-service release

**Date:** 2026-03-14
**Duration:** 47 minutes (09:12–09:59 UTC)
**Severity:** SEV-2
**Owner:** Payments team

## Summary

Roughly 38% of checkout attempts failed with `PAYMENT_FAILED` after release
`payment-service` v2.14.0 was promoted to 100% of prod traffic. The root cause
was a connection pool exhaustion caused by a new retry loop that opened a fresh
database connection on every retry.

## Timeline (UTC)

- 08:55 – v2.14.0 canary starts at 10% traffic. Error rate 0.3%, within threshold.
- 09:10 – Canary passes, rollout continues to 100%.
- 09:12 – Alert `PaymentFailureRateHigh` fires (failure rate > 5%).
- 09:18 – On-call engineer sees `FATAL: sorry, too many clients already` in
  payment-service logs.
- 09:31 – Decision to roll back. `kubectl argo rollouts undo payment-service
  -n payments` executed.
- 09:40 – Old version fully serving; failure rate dropping.
- 09:59 – Failure rate back to baseline 0.2%. Incident closed.

## Root cause

v2.14.0 added a retry around Stripe calls. The retry helper created a new
SQLAlchemy engine inside the loop instead of reusing the module-level one, so
every retry consumed an extra PgBouncer connection. At 10% traffic this stayed
under the 200-connection Cloud SQL limit; at 100% it exceeded it within two
minutes.

## Why the canary did not catch it

The canary threshold checks error rate and p99 latency, but the connection leak
only became fatal at full traffic volume. Connection count was not a canary
metric.

## Action items

1. Add `pg_stat_activity` connection count as a canary metric for
   payment-service. (Owner: Payments, done 2026-03-20)
2. Lint rule to forbid `create_engine` outside `db.py`. (Owner: Platform, done)
3. Extend canary duration from 15 to 30 minutes. (Owner: Payments, done)
4. Runbook update in troubleshooting.md. (Owner: Payments, done)

## Customer impact

Approximately 1,900 orders ended in `PAYMENT_FAILED`. notification-service sent
an apology email with a retry link on 2026-03-14 at 12:00 UTC. 71% of affected
customers completed their order within 24 hours.
