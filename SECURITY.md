# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 1.0.x | ✅ |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Open a [GitHub Issue](https://github.com/wesleyscholl/squish/issues) marked with
the `security` label, or email the maintainer directly (address on the GitHub
profile).

Include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested mitigations

You can expect an acknowledgement within 72 hours and a fix or mitigation plan
within 14 days for confirmed issues.

## Scope

Squish runs fully locally with no network traffic except:
- `huggingface-hub` model downloads (initiated explicitly by the user via `squish pull`)
- The API server listens on `localhost` only by default

There is no authentication on the local API server by design — it is intended
for single-user local use only.  Do not expose the server port to untrusted
networks without adding your own authentication layer (e.g. via a reverse proxy).
