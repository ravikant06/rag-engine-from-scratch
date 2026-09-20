# Database Guide

## Overview

Each service owns its own database. No service reads another service's tables
directly; cross-service data flows through Kafka events or HTTP APIs.

| Service | Database | Engine | Instance |
|---|---|---|---|
| order-service | orders | Cloud SQL for PostgreSQL 16 | acme-prod-orders (4 vCPU, 16 GB) |
| payment-service | payments | Cloud SQL for PostgreSQL 16 | acme-prod-payments (2 vCPU, 8 GB) |
| inventory-service | inventory | Cloud SQL for PostgreSQL 16 | acme-prod-inventory (2 vCPU, 8 GB) |
| notification-service | none (stateless) | – | – |

## Key tables

### orders.orders
Columns: `id` (UUID), `customer_id`, `status`, `total_cents`, `currency`,
`created_at`, `updated_at`. Status values: `CREATED`, `RESERVED`, `PAID`,
`PAYMENT_FAILED`, `SHIPPED`, `CANCELLED`.

### payments.payment_attempts
Columns: `id`, `order_id`, `stripe_charge_id`, `amount_cents`, `status`,
`failure_reason`, `attempted_at`. One order can have several attempts; the
latest attempt is the authoritative one.

### inventory.stock_levels
Columns: `sku`, `warehouse_id`, `available`, `reserved`, `updated_at`. Primary
key is (`sku`, `warehouse_id`).

## Migrations

We use Alembic for the Python services and golang-migrate for
inventory-service. Migrations run as a Kubernetes Job before the new
application version is rolled out. A migration must be backwards compatible
with the currently running version, because during a rolling update both old
and new pods talk to the same database.

## Backups

Cloud SQL automated backups run daily at 02:00 UTC with 14-day retention.
Point-in-time recovery is enabled on the `orders` and `payments` instances
only. Restore drills are performed quarterly by the Platform team.

## Connection pooling

Every service uses PgBouncer as a sidecar in transaction pooling mode. The
application pool size is 10 connections per pod. Direct connections to Cloud
SQL are limited to 200 per instance, so do not raise the pool size without
talking to the Platform team.
