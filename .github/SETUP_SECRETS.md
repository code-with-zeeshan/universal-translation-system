# Repository Secrets Setup

Configure the following secrets for CI publishing:

- PYPI_TOKEN: PyPI token with upload permissions for universal-decoder-node.
  - Create on https://pypi.org/manage/account/token/
  - Scope: Project-scoped (preferred) or Entire account (less preferred)

- NPM_TOKEN: npm access token with publish rights for both web and RN packages.
  - Create on https://www.npmjs.com/settings/<org-or-user>/tokens
  - Type: Automation (recommended)

How to add secrets:
1. GitHub → Settings → Secrets and variables → Actions → New repository secret
2. Name: PYPI_TOKEN, Value: <your-pypi-token>
3. Name: NPM_TOKEN, Value: <your-npm-token>

Least-privilege guidance:
- Prefer organization-level secrets with environment protection if multiple repos.
- Consider environments (staging, production) with required reviewers.