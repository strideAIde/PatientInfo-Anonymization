# CLAUDE.md — PatientInfo-Anonymization

This file provides guidance for AI assistants (Claude and others) working in this repository. It captures project intent, conventions, and development workflows.

---

## Project Overview

**PatientInfo-Anonymization** is a tool for de-identifying and anonymizing patient health information (PHI/PII) to support privacy compliance (HIPAA, GDPR, etc.) and safe data sharing for research or analytics.

> **Status:** Early scaffold — no implementation exists yet. This file documents the intended architecture and conventions to follow when building the project.

---

## Repository State

| Item | Status |
|------|--------|
| Source code | Not yet implemented |
| Tests | Not yet implemented |
| CI/CD | Not configured |
| Dependencies | Not declared |
| Docker / deployment | Not configured |

The only file tracked so far is `README.md` (placeholder). All implementation work should begin on a feature branch and follow the conventions below.

---

## Intended Architecture (to be implemented)

When building this project, follow this structure:

```
PatientInfo-Anonymization/
├── src/
│   ├── anonymizer/          # Core anonymization logic
│   │   ├── __init__.py
│   │   ├── engine.py        # Main anonymization pipeline
│   │   ├── detectors.py     # PHI/PII detection (NER, regex, etc.)
│   │   └── strategies.py    # Redaction, masking, pseudonymization
│   ├── parsers/             # Input format parsers (HL7, FHIR, CSV, plain text)
│   ├── exporters/           # Output format writers
│   └── config.py            # App-wide configuration
├── tests/
│   ├── unit/
│   └── integration/
├── data/
│   └── sample/              # Sample (non-real) test data only
├── docs/
├── .env.example
├── requirements.txt         # or pyproject.toml
├── Dockerfile
├── README.md
└── CLAUDE.md
```

---

## Technology Decisions (to be confirmed)

**Preferred stack (suggest to maintainer if not yet chosen):**

- **Language:** Python 3.11+
- **NLP / PHI detection:** spaCy with a medical NER model, or Microsoft Presidio
- **Data formats:** HL7 v2/v3, FHIR JSON, CSV, plain text
- **Testing:** `pytest` with `pytest-cov`
- **Linting / formatting:** `ruff` (linting + formatting), `mypy` (type checking)
- **Dependency management:** `pip` + `requirements.txt` or `poetry`/`pyproject.toml`

---

## Development Workflows

### Setting Up (once dependencies are declared)

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
# or: pip install -e ".[dev]"
```

### Running the Tool (once implemented)

```bash
python -m src.anonymizer.engine --input data/sample/patient.txt --output out/anonymized.txt
```

### Running Tests

```bash
pytest                        # run all tests
pytest --cov=src --cov-report=term-missing   # with coverage
pytest tests/unit/            # unit tests only
```

### Linting & Formatting

```bash
ruff check .                  # lint
ruff format .                 # format
mypy src/                     # type check
```

---

## Git Conventions

### Branching

| Branch | Purpose |
|--------|---------|
| `master` | Stable, reviewed code only |
| `krish-dev` | Active development branch |
| `claude/<id>` | AI-generated changes (opened as PRs, not merged directly) |

- All work should be done on a feature branch, never directly on `master`.
- Branch naming: `feature/<short-description>`, `fix/<short-description>`, `chore/<short-description>`.

### Commit Messages

Use the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <short summary>

[optional body]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`

Examples:
```
feat(detectors): add regex-based date detection for PHI
fix(engine): handle empty input files without crashing
test(detectors): add unit tests for SSN pattern matching
docs(readme): add installation instructions
```

---

## Security & Privacy Requirements

This project handles sensitive healthcare data. Adhere strictly to:

1. **No real patient data in the repository.** Use only synthetic or clearly fake test data.
2. **No PHI/PII in logs.** Logging should record events, not data values.
3. **No secrets in source.** Use `.env` files (gitignored) and `.env.example` for documentation.
4. **Dependency scanning.** Keep dependencies up to date; flag known vulnerabilities.
5. **Compliance considerations:** Design with HIPAA Safe Harbor and Expert Determination methods in mind.

---

## Code Conventions

### General

- Prefer clarity over cleverness.
- Functions should do one thing and be testable in isolation.
- All public functions/classes should have type annotations.
- Avoid mutable global state.

### PHI Anonymization Specifics

- Support at minimum these 18 HIPAA Safe Harbor identifiers: names, geographic data, dates, phone numbers, fax numbers, email addresses, SSNs, medical record numbers, health plan beneficiary numbers, account numbers, certificate/license numbers, vehicle identifiers, device identifiers, URLs, IP addresses, biometric identifiers, full-face photos, and other unique identifiers.
- Make anonymization strategies configurable (redact, mask, pseudonymize, generalize, suppress).
- Always return a report of what was detected and transformed.

### Error Handling

- Raise specific exceptions (not bare `Exception`).
- Never silently swallow errors involving data transformation — log and re-raise.

---

## AI Assistant Guidelines

When working in this repository as an AI assistant:

1. **Do not invent or generate synthetic patient data** that looks realistic (e.g., real-sounding SSNs, real-sounding names paired with medical conditions). Use obviously fake placeholders like `PATIENT_NAME`, `123-45-6789`, or clearly labeled test fixtures.
2. **Follow the architecture above** when adding new files — do not create ad-hoc scripts in the repo root.
3. **Write tests for every new feature.** No feature code should be merged without corresponding tests.
4. **Run linting and tests before committing.** If they cannot be run (e.g., dependencies not installed), note this explicitly in the commit message or PR description.
5. **Do not commit `.env` files** or any file containing credentials, API keys, or real patient data.
6. **Prefer small, focused PRs.** One logical change per PR.
7. **Leave `TODO:` comments** for decisions that require human review (e.g., choice of anonymization strategy for a specific field type).

---

## Contact / Maintainers

- **Repo owner:** Krishna-Prasad2001
- **Remote:** `http://local_proxy@127.0.0.1:37993/git/strideAIde/PatientInfo-Anonymization`

---

*Last updated: 2026-03-12*
