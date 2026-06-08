# Repository Secrets Setup

Configure the following secrets for CI publishing:

- PYPI_TOKEN: PyPI token with upload permissions for universal-decoder-node.
  - Create on https://pypi.org/manage/account/token/
  - Scope: Project-scoped (preferred)

- NPM_TOKEN: npm access token with publish rights for web and RN packages.
  - Create on https://www.npmjs.com/settings/<org-or-user>/tokens
  - Type: Automation (recommended)

- HF_TOKEN: Hugging Face access token for artifact uploads.
  - Create on https://huggingface.co/settings/tokens
  - Scope: write (or fine-grained per-repo)

Secrets used by workflows that must be set as Actions secrets or variables:

| Secret | Used by | Required |
|---|---|---|
| `PYPI_TOKEN` | publish-pypi.yml | For PyPI publishing |
| `NPM_TOKEN` | build-upload.yml | For npm publishing |
| `HF_TOKEN` | build-upload.yml | For HF Hub artifact uploads |
| `DOCKER_USERNAME` / `DOCKER_PASSWORD` | docker-build-publish.yml | For GHCR/DockerHub container push |
| `KUBECONFIG_B64` | kustomize-deploy.yml | For k8s deploy (env: production) |

How to add secrets:
1. GitHub -> Settings -> Secrets and variables -> Actions -> New repository secret
2. Name the secret as shown above, paste the value

Least-privilege guidance:
- Prefer organization-level secrets with environment protection.
- Consider environments (staging, production) with required reviewers.
- Map deployment workflows to environment: production.
- For local runtime secrets (HMAC keys, JWT secrets, API keys), see `docs/SECRET_MANAGEMENT.md`.
