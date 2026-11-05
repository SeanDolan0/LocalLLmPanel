# Security Policy

## Supported versions

Security fixes are generally developed against the latest version on the `main` branch. When reporting a vulnerability, include the affected application version and backend where relevant.

## Reporting a vulnerability

Please do not report exploitable vulnerabilities in a public issue.

Use GitHub's **Private vulnerability reporting** feature for this repository (Security → Advisories → Report a vulnerability) when it is enabled. If private vulnerability reporting is unavailable, contact the repository owner privately through GitHub before sharing technical details publicly.

Include the affected version, impact, reproduction steps, and any relevant logs or proof of concept. Redact API keys, Hugging Face/GitHub tokens, personal data, and private filesystem paths.

## Scope notes

The application coordinates local inference processes and exposes an optional OpenAI-compatible gateway. The gateway and managed servers are intended to bind to localhost by default; LAN exposure is an explicit user opt-in. Issues involving unintended network exposure, credential disclosure, unsafe process execution, or bypassing gateway authentication are particularly important to report privately.
