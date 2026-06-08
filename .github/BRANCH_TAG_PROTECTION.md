# Branch & Tag Protection Recommendations

## Branch protection (main)
- Require pull request before merging
- Require status checks to pass (CI workflows)
- Require linear history (optional)
- Require code owners review (optional)
- Restrict who can push to matching branches

## Tag protection
- Protect tags matching:
  - v*  (Core + Web SDK releases)
  - android-v* (Android SDK releases)
  - ios-v* (iOS SDK releases)
  - flutter-v* (Flutter SDK releases)
  - rn-v* (React Native SDK releases)
  - udn-v* (UDN decoder-node PyPI releases)
- Restrict creation to maintainers/release automation only

## Environments with approvals
- Create environments: staging, production
- Protect production with required reviewers
- Map deployment workflows to environment: production
- Store cluster credentials as environment-level secrets (e.g., KUBECONFIG_B64)