# Security policy

## Supported versions

| Version | Security updates |
|---|---|
| `1.3.x` | Supported |
| Earlier versions | Not supported |

The smart contracts and chain integration are optional research prototypes.
The confirmatory benchmark uses local off-chain evidence and does not require
contract deployment or a funded wallet.

## Reporting a vulnerability

Please do not disclose a suspected vulnerability in a public issue. Use
GitHub's private vulnerability-reporting feature on the repository **Security**
tab. If that feature is unavailable, contact the repository owner through a
private channel and include:

- the affected version or commit;
- the component and configuration;
- a minimal reproduction;
- the expected security impact; and
- any suggested mitigation.

Remove credentials, personal data, private infrastructure details, and live
endpoints from the report. Maintainers will acknowledge a complete report,
assess severity, coordinate a fix, and publish a security advisory when
appropriate.

## Operational guidance

- Copy `.env.example` or `.env.prod.example`; never commit the populated file.
- Generate unique API keys and use an external secret store in production.
- Keep chain signing keys outside the repository and application logs.
- Bind development services to localhost unless remote access is explicitly
  secured.
- Treat institutional-retrieval documents and uploaded data as untrusted
  input.
- Review dependency-audit output and GitHub security alerts before release.
