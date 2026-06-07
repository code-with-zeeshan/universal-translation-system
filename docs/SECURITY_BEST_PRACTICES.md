# Security Best Practices

This document outlines security best practices for deploying and operating the Universal Translation System in production.

## Table of Contents
1. [Secret Management](#secret-management)
2. [Authentication and Authorization](#authentication-and-authorization)
3. [Network Security](#network-security)
4. [Container Security](#container-security)
5. [Data Protection](#data-protection)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Secure Development Practices](#secure-development-practices)

## Secret Management

### Environment Variables
- **Never commit secrets to version control**
- Use environment variables for all sensitive information
- Prefer `*_FILE` environment variables in containers and orchestrators
- Centralized bootstrap validates required secrets and fails fast

### Rotation & JWKS
- Use rotation CLI: `tools/rotate_secrets.py` (supports HS256 and RS256)
- RS256: maintain multiple public keys via `JWT_PUBLIC_KEY` or `JWT_PUBLIC_KEY_PATH` with `||` separator

### JWT Secrets
- Generate strong secrets: `openssl rand -hex 32`
- Rotate periodically (at least every 90 days)
- Use different secrets per environment

### Example: Kubernetes Secrets
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: translation-system-secrets
type: Opaque
data:
  decoder-jwt-secret: <base64>
  coordinator-jwt-secret: <base64>
  coordinator-secret: <base64>
  coordinator-token: <base64>
  internal-service-token: <base64>
```
- See `kubernetes/secrets.example.yaml` for the full template.

## Authentication and Authorization

### API Authentication
- Always use JWT authentication for admin endpoints
- Implement proper token validation
- Use short-lived tokens (1 hour or less)

### Service Authentication (Internal)
- Use `X-Internal-Auth: <INTERNAL_SERVICE_TOKEN>` for internal-only endpoints
- Rotate internal tokens periodically

### Client Authentication
- Coordinator `/api/decode` requires `X-API-Key` with per-client keys
- Rate-limit by client key; emit 429 when exceeded

### Role-Based Access Control
- Implement RBAC for administrative functions
- Principle of least privilege

## Network Security

### TLS/SSL
- Always use HTTPS for all API endpoints
- Use TLS 1.2 or higher
- See `udn/utils/https_middleware.py` for HTTPS enforcement

### Network Policies
- Use Kubernetes Network Policies to restrict traffic
- Allow only necessary communication paths

## Container Security

### Image Security
- Use minimal base images (Ubuntu 22.04 for encoder, Python slim for others)
- All Dockerfiles use non-root users (`adduser`)
- Multi-stage builds for slim images

### Runtime Security
- Set resource limits for all containers
- Use read-only file systems where possible
- Mount secrets as volumes, not environment variables

## Data Protection
- TLS for data in transit
- Encrypt sensitive data at rest
- Only embeddings sent to cloud (not raw text)

## Monitoring and Logging
- Monitor for unusual access patterns
- Set up alerts for authentication failures
- Do not log sensitive information
- Use centralized logging with sectioned log handlers

## Secure Development Practices
- Code reviews with security focus
- Static code analysis
- Regular dependency updates
- Security gates in CI/CD pipeline

## Additional Resources
- [OWASP Top Ten](https://owasp.org/www-project-top-ten/)
- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/overview/)
- [JWT Security Best Practices](https://auth0.com/blog/a-look-at-the-latest-draft-for-jwt-bcp/)
