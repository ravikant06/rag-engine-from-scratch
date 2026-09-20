# Public API Reference (v1)

Base URL: `https://api.acme-commerce.example/v1`

All requests require an `Authorization: Bearer <JWT>` header. JWTs are issued by
the identity provider (Auth0) and expire after 1 hour. api-gateway rejects
invalid or expired tokens with `401 Unauthorized`.

## Rate limits

100 requests per minute per customer. Exceeding the limit returns
`429 Too Many Requests` with a `Retry-After` header.

## Endpoints

### POST /orders
Create an order.

Request body:
```json
{
  "items": [{"sku": "SKU-123", "quantity": 2}],
  "payment_method_id": "pm_abc123",
  "currency": "EUR"
}
```
Responses:
- `201 Created` with the order object (status will be `CREATED` or `RESERVED`).
- `409 Conflict` if any item is out of stock. Body contains `out_of_stock_skus`.
- `422 Unprocessable Entity` for validation errors.

### GET /orders/{order_id}
Fetch one order, including its current `status` and `payment_status`.
Returns `404 Not Found` if the order does not belong to the caller.

### GET /orders
List the caller's orders, newest first. Supports `?status=` filter and
`?cursor=` pagination with a page size of 20.

### POST /orders/{order_id}/cancel
Cancel an order. Only allowed while status is `CREATED`, `RESERVED` or
`PAYMENT_FAILED`. Returns `409 Conflict` otherwise. Cancelling releases the
stock reservation in inventory-service.

## Idempotency

`POST /orders` accepts an optional `Idempotency-Key` header. Requests with the
same key within 24 hours return the original response instead of creating a
duplicate order. Keys are stored in the `orders.idempotency_keys` table.

## Versioning

Breaking changes get a new path prefix (`/v2`). Additive changes (new optional
fields) ship under the existing version without notice.
