# Platform Architecture Overview

Acme Commerce runs an order-processing platform made of four backend services
and one public gateway. All services are written in Python 3.11 (FastAPI) except
`inventory-service`, which is written in Go 1.22.

## Services

| Service | Responsibility | Owner team |
|---|---|---|
| api-gateway | Public HTTPS entry point, JWT validation, rate limiting | Platform |
| order-service | Creates and tracks orders, owns the `orders` database | Checkout |
| payment-service | Charges cards via Stripe, records payment attempts | Payments |
| inventory-service | Tracks stock levels per warehouse | Fulfilment |
| notification-service | Sends order emails and SMS via SendGrid and Twilio | Checkout |

## Request flow for a checkout

1. The web client calls `POST /v1/orders` on api-gateway.
2. api-gateway validates the JWT and forwards to order-service.
3. order-service reserves stock by calling inventory-service (`POST /reserve`).
4. order-service publishes an `order.created` event to the `orders` Kafka topic.
5. payment-service consumes `order.created`, charges the card, and publishes
   `payment.succeeded` or `payment.failed`.
6. order-service consumes the payment event and moves the order to `PAID` or
   `PAYMENT_FAILED`.
7. notification-service consumes `payment.succeeded` and emails the receipt.

## Messaging

We use a single Kafka cluster (3 brokers, managed by Confluent Cloud). Topics
have 12 partitions and 7-day retention. Every event carries an `order_id`
header, which is also the partition key so events for one order stay ordered.

## Synchronous vs asynchronous

Stock reservation is synchronous because the customer must be told immediately
if an item is out of stock. Payment and notifications are asynchronous because
they can take seconds and the customer only needs a confirmation that the order
was accepted.

## Environments

There are three environments: `dev`, `staging`, and `prod`. All three run on
Google Kubernetes Engine (GKE) in region `europe-west1`. See deployment.md for
how services get there.
