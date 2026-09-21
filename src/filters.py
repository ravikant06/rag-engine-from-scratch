"""
Build Qdrant filters from plain, user-facing parameters.

Two rules shape this module:

1. `tenant_id` is ALWAYS applied. A caller that forgets it gets
   config.TENANT_ID, never an unscoped search. Cross-tenant leakage is the
   one failure mode worth making structurally impossible.

2. Everything else is optional and defaults to None (no restriction). Each
   filter narrows the candidate set *before* the vector search runs, so
   top_k is filled from the matching subset rather than trimmed afterwards.
"""
from datetime import datetime, timezone

from qdrant_client.http import models

from src import config

# Filters a caller may pass. Anything else raises, so a typo like
# `doc_typ="incident"` fails loudly instead of silently matching everything.
ALLOWED_FILTERS = {
    "doc_type",
    "source",
    "doc_id",
    "service",
    "severity",
    "owner",
    "date_from",
    "date_to",
}

# payload field <- filter name, for the plain "match any of these" filters.
_MATCH_FIELDS = {
    "doc_type": "doc_type",
    "source": "source",
    "doc_id": "doc_id",
    "service": "services",   # list payload; MatchAny tests membership
    "severity": "severity",
    "owner": "owner",
}


def _as_list(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def _to_epoch(value: str) -> int:
    """'2026-03-01' -> epoch seconds (UTC midnight)."""
    try:
        return int(datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    except ValueError as exc:
        raise SystemExit(f"Dates must look like YYYY-MM-DD, got: {value!r}") from exc


def build_filter(tenant_id: str | None = None, **where) -> models.Filter:
    """
    Compose a Qdrant filter. `where` accepts any key in ALLOWED_FILTERS;
    string or list values both work:

        build_filter(tenant_id="acme", doc_type="incident")
        build_filter(service=["payment-service", "api-gateway"])
        build_filter(doc_type="incident", date_from="2026-03-01")
    """
    unknown = set(where) - ALLOWED_FILTERS
    if unknown:
        raise SystemExit(
            f"Unknown filter(s): {', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(sorted(ALLOWED_FILTERS))}"
        )

    # Tenant scope is not optional.
    conditions: list[models.FieldCondition] = [
        models.FieldCondition(
            key="tenant_id",
            match=models.MatchValue(value=tenant_id or config.TENANT_ID),
        )
    ]

    for name, field in _MATCH_FIELDS.items():
        value = where.get(name)
        if value:
            conditions.append(
                models.FieldCondition(key=field, match=models.MatchAny(any=_as_list(value)))
            )

    date_from, date_to = where.get("date_from"), where.get("date_to")
    if date_from or date_to:
        # Only dated documents (incidents) carry date_ts, so a date filter
        # deliberately excludes everything else.
        conditions.append(
            models.FieldCondition(
                key="date_ts",
                range=models.Range(
                    gte=_to_epoch(date_from) if date_from else None,
                    lte=_to_epoch(date_to) if date_to else None,
                ),
            )
        )

    return models.Filter(must=conditions)


def describe(tenant_id: str | None = None, **where) -> str:
    """Human-readable summary of the active filters, for printing."""
    parts = [f"tenant={tenant_id or config.TENANT_ID}"]
    parts += [f"{k}={v}" for k, v in sorted(where.items()) if v]
    return ", ".join(parts)
