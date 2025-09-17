# Autoscaling and Readiness

This project exposes health and readiness endpoints and includes example HPAs to enable safe scaling.

## Endpoints
- Decoder
  - GET /health → { status: "ok", version, apiVersion }
  - GET /ready → 200 when model+vocabulary are initialized and JWKS (if RS256) is available; else 503
- Coordinator
  - GET /health → { status: "ok", version, apiVersion }
  - GET /ready → 200 when JWKS (if RS256) is available; includes pool state; else 503

## Kubernetes Probes
- Decoder: liveness /health, readiness /ready
- Coordinator: liveness /health, readiness /ready

## Horizontal Pod Autoscalers
- decoder-hpa.yaml: scales 1–3 on CPU 70%
- coordinator-hpa.yaml: scales 1–5 on CPU 60%

Adjust targets based on actual usage (and consider Prometheus metrics for functional scaling).

## Metric-based Autoscaling (Optional)
For advanced scenarios, use Prometheus Adapter to expose custom metrics (e.g., request rate, queue depth).
- Export metrics (already available via /metrics, protected with JWT for coordinator)
- Configure Prometheus + Adapter
- Define HPA with `type: Pods` or `type: Object` referencing custom metric (e.g., `http_requests_per_second`)

## Safe Disruptions
- PodDisruptionBudgets ensure at least one pod stays available during voluntary disruptions.
- Update windows: prefer surge or rolling updates with maxUnavailable=0 for stateful decoder workloads.