# Security Policy

The Aura AI team takes security and data isolation seriously. This document outlines the security posture, vulnerability reporting process, and baseline protections enforced across the system.

---

## 1. Supported Versions

| Version | Supported | Security Maintenance |
| :--- | :---: | :--- |
| **1.0.x** | :white_check_mark: | Current production release |
| < 1.0 | :x: | Unsupported development builds |

---

## 2. Reporting a Vulnerability

If you discover a security vulnerability or weakness within Aura AI, please report it responsibly:

1. **Do NOT open a public issue** on GitHub for undisclosed security vulnerabilities.
2. **Contact**: The project maintainers via the designated security contact configured for this repository (e.g., via GitHub Private Vulnerability Reporting on the repository's Security tab). If no dedicated security email is configured in your deployment fork, please contact the repository owner directly through their private profile contact.
3. **Information to Include**:
   - Description of the vulnerability and attack vector.
   - Minimal reproducible test case or steps.
   - Impact assessment (e.g., potential for data leakage, unauthorized query execution).
   - Any suggested remediations or patches.

We appreciate your effort in responsibly disclosing issues and will work to review and address verified reports promptly.

---

## 3. Built-In Security Architecture

Aura AI implements layered security controls tested against an adversarial test suite:

- **Prompt Injection Boundaries**: Structured separation of trusted system prompts and untrusted document text.
- **Citation Fabrication Defense**: Unbacked citation markers (e.g., `[S99]`) are stripped automatically by the post-generation citation validator.
- **Session Data Isolation**: Cryptographic session separation ensuring Session A cannot read or influence Session B messages or citations.
- **Safe Text-to-SQL Sandbox**:
  - AST validation prohibiting DDL/DML, PRAGMA statements, ATTACH commands, and multiple statements.
  - SQLite low-level C-authorizer hook rejecting all write and modification operations.
  - Deterministic progress handler timeout halting runaway recursion at $5.0\text{ s}$.
  - Read-only database connection handles preserving data integrity.
- **Path Traversal Protection**: Filename sanitization preventing `../../` escape outside the designated upload directory.
- **Resource Exhaustion Defense**: File size bounded to 50MB; image decompression bounded to 50 megapixels.
- **Safe Error Handling**: Internal stack traces and file paths are masked from HTTP error payloads.
