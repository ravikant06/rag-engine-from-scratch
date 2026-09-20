# Troubleshooting Runbook

Start with the dashboards in Grafana (folder "Acme Commerce / Prod"), then use
the sections below. All commands assume `kubectl` is pointed at the prod
cluster (`gcloud container clusters get-credentials acme-prod --region europe-west1`).

## Payment failure rate is high

Alert: `PaymentFailureRateHigh` (fires when > 5% for 3 minutes).

1. Check whether a payment-service rollout is in progress:
   `kubectl argo rollouts get rollout payment-service -n payments`
2. Look for `too many clients already` in logs:
   `kubectl logs -n payments -l app=payment-service --since=10m | grep -i "too many"`
   If present, this is connection pool exhaustion (see incident-101.md). Roll
   back immediately.
3. Check Stripe status at status.stripe.com. If Stripe is degraded, enable the
   `payments.queue_on_provider_error` feature flag so orders wait instead of failing.
4. If neither applies, page the Payments on-call via PagerDuty.

## Orders stuck in RESERVED

Usually means payment-service is not consuming the `orders` Kafka topic.

1. Check consumer lag in Confluent Cloud for consumer group `payment-service`.
2. Restart the consumers: `kubectl rollout restart deployment payment-service -n payments`
   (this is safe; the consumer is idempotent on `order_id`).
3. If lag keeps growing, check whether the topic partition count changed
   recently. Partitions must remain at 12.

## 409 Conflict on POST /orders for items that are in stock

inventory-service may have stale `reserved` counts from cancelled orders.

1. Query: `SELECT sku, available, reserved FROM stock_levels WHERE sku = '...'`
2. Run the reconciliation job: `kubectl create job --from=cronjob/inventory-reconcile manual-reconcile -n fulfilment`
3. The job recomputes `reserved` from open orders. Takes about 2 minutes.

## 401 Unauthorized for valid users

1. Confirm Auth0 is healthy (status.auth0.com).
2. Check the JWKS cache on api-gateway. Cloud API Gateway caches keys for 5
   minutes; a key rotation can cause a short burst of 401s. Wait 5 minutes.

## Emails not sent

1. Check SendGrid activity feed for bounces.
2. Verify notification-service is consuming `payment.succeeded`.
3. notification-service has no database, so a restart loses nothing:
   `kubectl rollout restart deployment notification-service -n checkout`

## Getting help

- #acme-oncall in Slack for anything urgent.
- Platform team owns GKE, Kafka, Cloud SQL and CI/CD.
- Each service's owning team (see architecture.md) owns application bugs.
