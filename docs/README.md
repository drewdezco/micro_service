# Micro Service Documentation

Welcome! This docs section tracks our collaborative effort to build the first Python microservice for this repository.

- Project goal: Create a minimal, production-ready microservice with clear patterns we can reuse for future services.
- Primary contributors: Drew H and Tim L

## Contents

- [Overview](./overview.md) — architecture, components, decisions
- [Getting Started](./getting-started.md) — setup, environment, run/test
- [API Reference](./api.md) — endpoints, contracts, examples
- [Contributing](./contributing.md) — workflow, branches, code style

## Project Charter

- Scope: Single Python microservice exposing a simple HTTP API with one or two endpoints, health checks, and basic observability.
- Non-goals (for v0): Multi-service orchestration, complex auth, advanced CI/CD.
- Success criteria:
  - `GET /health` returns 200 with service metadata
  - One functional endpoint with input validation and tests
  - Local dev experience: `make dev` or single command to run
  - Lint, type check, and unit tests pass locally

## Tech Stack (proposed)

- Language: Python 3.11+
- Web framework: FastAPI (or Flask if preferred)
- Server: Uvicorn (for FastAPI)
- Package management: `uv` or `pip + venv` (team preference)
- Linting/formatting: Ruff + Black
- Type checking: MyPy
- Testing: Pytest
- Config: `.env` via `pydantic-settings` (FastAPI) or `python-dotenv`
- Make or Invoke tasks for common commands
- Optional: Dockerfile for consistent runtime

We can adapt any item above based on team consensus.

## Initial Milestones

1. Bootstrap
   - Choose framework (FastAPI recommended)
   - Scaffold project layout
   - Add health endpoint and versioning

2. Quality Gates
   - Add Ruff, Black, MyPy, Pytest
   - Add pre-commit hooks

3. Core Endpoint
   - Implement first functional endpoint (noun-verb pair)
   - Input/output schemas and pydantic models
   - Unit tests and example requests

4. Observability
   - Structured logging
   - Basic metrics counter/timer (optional)
   - Error handling and standardized error response format

5. Packaging & Run
   - Dockerfile (slim image)
   - Makefile tasks
   - README updates

## Proposed Repository Structure
micro_service/
docs/
README.md
overview.md
getting-started.md
api.md
contributing.md
src/
micro_service/
init.py
main.py # FastAPI app entrypoint
api/
init.py
routes.py # API routers
schemas.py # pydantic models
core/
config.py # settings
logging.py # logging config
services/
init.py
example.py # business logic
tests/
test_health.py
test_example.py
pyproject.toml
Makefile
.env.example
.gitignore


## Collaboration Plan

- Branching: feature branches off `main` (e.g., `feature/health-endpoint`, `chore/tooling-setup`)
- Reviews: Require 1 review (Drew H or Tim L) before merge
- Commits: Conventional commits style recommended (e.g., `feat: add health endpoint`)
- Issues: Use GitHub issues for tasks; link PRs to issues
- Standups: As needed; keep status in PR descriptions

## Ownership

- Lead developers: 
  - Drew H — scaffolding, API design, CI bootstrap
  - Tim L — tooling, tests, observability
- Shared responsibilities: docs, reviews, endpoint implementations

## Next Actions

- Decide FastAPI vs Flask
- Create scaffolding PR with:
  - minimal app + `GET /health`
  - pyproject with Black/Ruff/MyPy/Pytest
  - Makefile tasks
  - `.env.example`
  - initial tests
- Draft API contract in [API Reference](./api.md)

## Conventions

- Code style: Black + Ruff defaults
- Types: Use type hints throughout; MyPy clean
- Errors: Return JSON errors with `code`, `message`, `details`
- Logging: Structured logs with request ID, method, path, status

## Contact
Contact
Primary contributors: Drew H, Tim L
Open an issue in GitHub for questions, proposals, or bugs.


## Example Health Response

```json
{
  "status": "ok",
  "service": "micro_service",
  "version": "0.1.0",
  "timestamp": "2025-01-01T12:00:00Z"
}

