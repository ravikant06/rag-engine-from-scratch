# Deployment Guide

All Acme Commerce services are deployed as containers to GKE using Helm charts
stored in the `acme/infra-charts` repository. CI/CD is GitHub Actions.

## Pipeline stages

1. **Build** – On every push, GitHub Actions builds a Docker image tagged with
   the short git SHA and pushes it to Artifact Registry
   (`europe-west1-docker.pkg.dev/acme-prod/services/<service>`).
2. **Test** – Unit tests and contract tests run against the built image. A
   failing test blocks the pipeline.
3. **Deploy to dev** – Automatic on merge to `main`.
4. **Deploy to staging** – Automatic once dev smoke tests pass.
5. **Deploy to prod** – Manual approval required from a member of the owning
   team. Approval is given in the GitHub Actions UI.

## payment-service specifics

payment-service is the only service deployed with a **canary strategy**. A new
version first receives 10% of traffic for 15 minutes. If the error rate stays
below 0.5% and p99 latency below 800 ms, the rollout continues to 100%.
Otherwise the canary is rolled back automatically by Argo Rollouts.

payment-service runs 6 replicas in prod (minimum 4, maximum 12 under HPA based
on CPU at 60%). Each pod requests 500m CPU and 1 GiB memory.

## Other services

order-service, inventory-service and notification-service use a plain
RollingUpdate with `maxSurge: 1` and `maxUnavailable: 0`. order-service runs 4
replicas in prod; inventory-service and notification-service run 3 each.

api-gateway is not a Kubernetes deployment at all: it is Google Cloud API
Gateway configured from an OpenAPI spec in `acme/api-gateway-config`.

## Rollback

To roll back any service, re-run the "Deploy to prod" job for the previous
successful commit. For payment-service, `kubectl argo rollouts undo
payment-service -n payments` is faster.

## Secrets

Secrets (Stripe keys, database passwords, Twilio tokens) live in Google Secret
Manager and are mounted into pods with the External Secrets Operator. Never put
secrets in Helm values files.
