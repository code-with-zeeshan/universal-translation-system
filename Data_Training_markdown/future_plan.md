# Future Plan

This document outlines the future roadmap for the Universal Translation System, with a focus on scalability, automation, and analytics.

## 1. Dynamic Scaling
- Use the advanced coordinator to dynamically add/remove decoder nodes at runtime
- Support zero-downtime upgrades and rolling deployments
- Integrate with cloud auto-scaling (Kubernetes, AWS, GCP, Azure)

## 2. Config-Driven Pipeline
- All language, data, and training configs are centralized in `data/config.yaml` and `config/training_*.yaml`
- Pipeline is orchestrated by `data/practical_data_pipeline.py` for reproducibility

## 3. Monitoring & Analytics
- Use Prometheus and Grafana for real-time monitoring of data, training, and inference
- Advanced coordinator dashboard provides live analytics, charts, and admin controls
- Track per-decoder uptime, load, and error rates

## 4. SDK & API Evolution
- Continue to expand SDK support (Flutter, React Native, Web)
- Add new language pairs and improve integration tests
- Support WASM/edge decoding for web/JS in the future

## 5. Automation & CI/CD
- Automate model builds, tests, and deployments using GitHub Actions and Kubernetes Jobs
- Use the coordinator API for automated node registration and health checks

## 6. Advanced Dashboard Features
- Authentication UI, real-time charts, advanced analytics, and manual routing for admin users
- Role-based access and historical analytics (planned)

---

For current architecture and dashboard features, see [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).