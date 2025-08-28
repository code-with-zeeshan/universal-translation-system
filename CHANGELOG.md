# Changelog

All notable changes to the Universal Translation System will be documented in this file.

## [Unreleased]

### Added
- Environment variable configuration for all components
- Docker and Kubernetes deployment support
- Comprehensive Prometheus/Grafana monitoring dashboards
- New VISION.md document explaining the system architecture and goals
- Enhanced train_from_scratch.py script with improved command-line interface
- Reorganized documentation structure for better navigation
- Comprehensive Prometheus configuration with alerting and recording rules
- Security best practices documentation
- Kubernetes health probes and resource requests
- Improved Docker health checks using wget instead of curl
- SDK_PUBLISHING.md guide (Android Maven, iOS Podspec/SPM, RN linking)
- Web example Express server with proper WASM headers (COOP/COEP/CORS)
- README updates for Android/iOS/Flutter/Web SDKs with coordinator usage
- GitHub Actions workflows: sdk-publish.yml (Android/iOS) and web-npm-publish.yml (web)
- Coordinator periodic Redis-to-disk mirroring via `COORDINATOR_MIRROR_INTERVAL` (min 5s, logs effective value)

### Changed
- Updated README.md with current system capabilities and Docker deployment instructions
- Improved CONTRIBUTING.md with Docker deployment instructions and better contributor guidelines
- Enhanced FAQ.md to reflect current system capabilities and remove outdated references
- Completely rewrote Vocabulary_Guide.md to focus on the dynamic vocabulary system
- Updated future_plan.md with a forward-looking roadmap and moved to docs folder
- Added environment variable configuration section to monitoring/README.md
- Improved .gitignore with more comprehensive patterns for various development environments
- Moved Adding_New_languages.md to docs folder with updated instructions
- Updated Quick Start guide in README.md with more comprehensive options
- Coordinator now uses `RedisManager` for sync access in async paths and mirrors Redis to disk in reload/save flows

### Removed
- Outdated GOAL.md file (replaced by docs/VISION.md)
- References to non-existent files in documentation
- Redundant files from Data_Training_markdown folder
- Legacy analysis reports from report folder

## [0.1.0] - 2025-08-22

### Added
- Initial release of Universal Translation System
- Edge encoding, cloud decoding architecture
- Support for 20 languages with dynamic vocabulary loading
- Native SDKs for Android, iOS, Flutter, React Native, and Web
- Basic monitoring with Prometheus metrics
- Coordinator for load balancing and health monitoring