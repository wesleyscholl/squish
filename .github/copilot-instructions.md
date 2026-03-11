# CLAUDE.md — Project Conventions & AI Collaboration Guidelines

This file defines standing instructions for Claude when working on any project in this repository. Always read and follow these rules before writing, modifying, or deleting any code or documentation.

---

## 🗂️ Planning First

- **Always read `PLAN.md`, `ROADMAP.md`, or equivalent planning docs before starting any task.**
- Identify the relevant phase, step, or milestone before writing or modifying any code.
- If no plan exists, create one before proceeding and ask for confirmation.
- After completing work, update `PLAN.md`, `ROADMAP.md`, `README.md`, and any relevant docs to reflect what changed, what's done, and what's next.
- If a task deviates from the current plan, call it out explicitly before continuing.

---

## 🧱 Code Quality

- **All written code must be production-grade at all times.** No placeholders, no "good enough for now," no TODOs left in shipped code.
- Follow the language's idiomatic style and best practices (e.g., Rust ownership patterns, Python type hints, Go error handling).
- Prefer explicit over implicit. Avoid magic values — use named constants.
- Keep functions small, focused, and single-responsibility.
- Handle all error cases. Never silently swallow errors.
- Avoid code duplication. Extract shared logic into reusable utilities or modules.
- Add inline comments only where intent is non-obvious. Code should be self-documenting first.

---

## 🧪 Testing

- **100% test coverage is required.** Every code file must have a corresponding test file.
- Tests live alongside their source file or in a dedicated `tests/` directory — be consistent with the project's convention.
- Write unit tests for all functions, integration tests for module interactions, and end-to-end tests where applicable.
- Tests must be deterministic. No flaky tests, no time-dependent assertions without mocking.
- All tests must pass before committing. Never commit with known failing tests.
- Use table-driven or parameterized tests for functions with multiple input/output cases.
- Test edge cases, error paths, and boundary conditions — not just the happy path.

---

## 🖥️ Command Output

- **Never suppress command output.** Do not use `2>/dev/null`, `2>&1`, `--quiet`, `-q`, or any flag that hides stderr/stdout unless the user explicitly requests it.
- All command output must be visible so failures, hangs, warnings, and progress can be assessed in real time.
- If a command is long-running, prefer flags that show progress (e.g., `--verbose`, `-v`) where available.
- If a command hangs, report it — do not silently wait indefinitely.

---

## 🔁 Git Workflow

- **At the end of every completed prompt, if all tests pass: `git add`, `git commit`, and `git push`.**
- Write clear, imperative commit messages that describe *what* and *why*:
  - ✅ `feat(inference): add KV cache eviction strategy for M3 memory pressure`
  - ❌ `updates`, `fix stuff`, `WIP`
- Follow [Conventional Commits](https://www.conventionalcommits.org/) format: `type(scope): description`
  - Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `ci`
- Never force-push to `main` or `master` without explicit instruction.
- Commit logical units of work — don't bundle unrelated changes in one commit.

---

## 📁 File & Project Structure

- Follow the existing project structure. Do not reorganize directories without discussing it first.
- New modules, crates, or packages should match the naming convention already in use.
- Keep config, secrets, and environment-specific values out of source code. Use `.env` files or environment variables, and ensure `.gitignore` is up to date.
- Delete dead code. Don't comment it out and leave it — use version control for history.

---

## 📝 Documentation

- Update `README.md` whenever public-facing behavior, setup steps, or architecture changes.
- Update `PLAN.md` / `ROADMAP.md` after every session — mark completed items, add new findings, note blockers.
- Document all public APIs, functions, and types with docstrings/doc comments appropriate to the language.
- If a non-obvious architectural decision is made, add an `ADR` (Architecture Decision Record) entry or a note in the relevant doc.

---

## 🔍 Before You Build

Before writing any code, confirm:
1. ✅ You've read the relevant plan/phase/step.
2. ✅ You understand what "done" looks like for this task.
3. ✅ You know where the new code lives in the project structure.
4. ✅ You know what tests will prove it works.
5. ✅ You know what docs need updating when it's done.

If any of these are unclear, ask before proceeding.

---

## 🚫 Hard Stops

Do not proceed if:
- Tests are failing from a previous step (fix them first).
- The plan is ambiguous or missing for a non-trivial task.
- A change would break a public API or interface without a migration path.
- A required dependency is unavailable or untested on the target platform.

---

## 🧠 General Mindset

- Prefer boring, correct, and maintainable over clever.
- When in doubt, do less and confirm — don't assume scope.
- Surface trade-offs and alternatives when they exist; don't just pick one silently.
- Treat every PR as if a senior engineer on the team will review it.