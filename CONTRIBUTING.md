# Contributing to Aura AI

Thank you for your interest in contributing to Aura AI! We welcome contributions that maintain or enhance the system's reliability, retrieval performance, and safety.

---

## 1. Development Setup

1. **Prerequisites**:
   - Python 3.11
   - Virtual environment manager (`venv`)
   - Git

2. **Setup Instructions**:
   ```bash
   # Clone the repository
   git clone https://github.com/your-username/aura-ai.git
   cd aura-ai

   # Create and activate virtual environment
   python -m venv .venv
   # Windows:
   .venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate

   # Install development dependencies
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install pytest
   ```

3. **Configure Environment**:
   ```bash
   cp .env.example .env
   ```
   Set your local development parameters in `.env`.

---

## 2. Testing Requirements

Aura AI maintains a 100% green regression test baseline. Any code modification must preserve all existing tests.

```bash
# Run the full regression test suite
pytest -q
```
- **Requirement**: All 204 existing unit and integration tests must pass with 0 failures before any pull request is considered.
- **New Tests**: Any new feature or bug fix must include corresponding deterministic tests in the `tests/` directory.

---

## 3. Coding Expectations

- **Code Quality**: Follow PEP 8 style conventions. Keep functions modular and type-annotated where appropriate.
- **Architectural Boundaries**:
  - Do not modify core data contracts in `app/models.py` without backward-compatibility considerations.
  - Keep configuration centralized in `app/config.py`. Never hardcode timeouts, candidate limits, or filesystem paths.
  - Preserve the separation of concerns between `app/logic.py` (retrieval/processing), `app/api.py` (HTTP endpoints), `app/sessions.py` (SQLite persistence), and `app/sql_engine.py` (structured execution).
- **Documentation**: If a change modifies API behavior or parameters, update corresponding documentation in `README.md` and docstrings.

---

## 4. Security & Safety Principles

- **No Secrets in Code**: Never commit real API keys, credentials, or private data to version control.
- **Maintain Defenses**: Never weaken or bypass existing security controls:
  - AST SQL validation and SQLite C-authorizer hooks.
  - SQL execution timeouts.
  - Path traversal protections and filename sanitization.
  - Image decompression bomb checks.
  - Prompt boundary defenses and citation validation.

---

## 5. Reporting Security Vulnerabilities

If you discover a potential security vulnerability, please refer to [SECURITY.md](SECURITY.md) for responsible disclosure guidelines.
