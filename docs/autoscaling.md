# Autoscaling and Readiness

This project exposes health and readiness endpoints and includes example HPAs to enable safe scaling.

## Endpoints
- Decoder
  - GET /health -> { status: "ok", version, apiVersion }
  - GET /ready -> 200 when model+vocabulary initialized and JWKS (if RS256) available; else 503
- Coordinator
  - GET /health -> { status: "ok", version, apiVersion }
  - GET /ready -> 200 when JWKS available; includes pool state; else 503

## Kubernetes Probes
- Decoder: liveness /health, readiness /ready
- Coordinator: liveness /health, readiness /ready

## Horizontal Pod Autoscalers
- decoder-hpa.yaml: scales 1-3 on CPU 70%
- coordinator-hpa.yaml: scales 1-5 on CPU 60%

## Helm Chart
The Helm chart at `charts/uts/` includes HPA configurations and resource requests/limits for all services.

## Safe Disruptions
- PodDisruptionBudgets ensure at least one pod stays available during voluntary disruptions.
- Rolling updates with maxUnavailable=0 for stateful decoder workloads.
