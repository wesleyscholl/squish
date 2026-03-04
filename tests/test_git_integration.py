"""
tests/test_git_integration.py

Integration tests — require a live or auto-startable Squish server.

These tests cover two paths through the commit-message generation flow:

  1. `squish git --no-commit <path>`  — the native Squish CLI path
     (preferred; richer prompt + structured cleanup).

  2. `squish run` (piped prompt)      — the path used by git-commit-push-script.sh:
         echo "$PROMPT" | squish run --max-tokens 60 --temperature 0.2

Select a model with --model or $SQUISH_MODEL:
    pytest tests/test_git_integration.py -v --model 14b
    SQUISH_MODEL=14b pytest tests/test_git_integration.py -v

Skip entirely:
    SQUISH_SKIP_INTEGRATION=1 pytest tests/test_git_integration.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

SQUISH_ROOT = Path(__file__).resolve().parent.parent
CLI         = str(SQUISH_ROOT / "squish" / "cli.py")

# Shell-script constants — must stay in sync with git-commit-push-script.sh
_SCRIPT_TIMEOUT_SECONDS  = 45
_SCRIPT_MAX_COMMIT_LEN   = 50


# ── helpers ───────────────────────────────────────────────────────────────────

def _squish(*cmd_args, input_text: str | None = None, timeout: int = 120):
    """Run  python3 cli.py <cmd_args>  and return CompletedProcess."""
    return subprocess.run(
        [sys.executable, CLI, *cmd_args],
        capture_output=True,
        text=True,
        input=input_text,
        timeout=timeout,
    )


def _make_staged_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo with one staged change and return its path."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)

    git("init")
    git("config", "user.email", "test@squish.ai")
    git("config", "user.name",  "Squish Test")

    # Initial commit so HEAD exists
    (repo / "README.md").write_text("# Test repo\n")
    git("add", "README.md")
    git("commit", "-m", "init")

    # Staged change to generate a message for
    (repo / "utils.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def subtract(a, b):\n"
        "    return a - b\n\n"
        "def multiply(a, b):\n"
        "    return a * b\n"
    )
    git("add", "utils.py")
    return repo


def _extract_commit_message(stdout: str) -> str:
    """
    Pull the commit message out of `squish git --no-commit` stdout.
    The CLI prints it as:  '  <bold-escape>MESSAGE<reset-escape>\n'
    We strip ANSI codes and status-line prefixes to isolate the message.
    """
    import re
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    lines = [ansi_escape.sub("", l).strip() for l in stdout.splitlines()]
    # Skip status lines that start with known prefixes
    skip_prefixes = ("▸", "✓", "→", "[", "squish", "Server", "waiting", ".")
    msg_lines = [l for l in lines if l and not any(l.startswith(p) for p in skip_prefixes)]
    return msg_lines[0] if msg_lines else ""


# ── pytest configuration ──────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption(
        "--model", default=None,
        help="Model hint passed to squish --model  (e.g. '14b', '7b', full path)"
    )


@pytest.fixture(scope="session")
def model_hint(pytestconfig):
    """Model hint from --model pytest flag or $SQUISH_MODEL env var."""
    return pytestconfig.getoption("--model") or os.environ.get("SQUISH_MODEL")


@pytest.fixture(scope="session", autouse=True)
def _skip_if_disabled():
    if os.environ.get("SQUISH_SKIP_INTEGRATION"):
        pytest.skip("SQUISH_SKIP_INTEGRATION is set — skipping integration tests")


@pytest.fixture(scope="session")
def staged_repo(tmp_path_factory):
    """A git repo with one staged change, shared across the session."""
    return _make_staged_repo(tmp_path_factory.mktemp("git"))


# ── Test class: squish git --no-commit ───────────────────────────────────────

class TestSquishGitCommand:
    """
    Tests for `squish git --no-commit <path>`.

    This is the preferred path for AI-generated commit messages:
    richer prompt, structured cleanup, Jira ticket detection.
    """

    def test_generates_non_empty_message(self, staged_repo, model_hint):
        """squish git should produce a non-empty commit message."""
        cmd = ["git", "--no-commit", str(staged_repo)]
        if model_hint:
            cmd = ["--model", model_hint, *cmd]

        result = _squish(*cmd, timeout=120)

        assert result.returncode == 0, (
            f"squish git exited {result.returncode}\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )
        msg = _extract_commit_message(result.stdout)
        assert msg, f"No commit message found in output:\n{result.stdout}"

    def test_message_length(self, staged_repo, model_hint):
        """Commit message must not exceed 72 characters (CLI trims at word boundary)."""
        cmd = ["git", "--no-commit", str(staged_repo)]
        if model_hint:
            cmd = ["--model", model_hint, *cmd]

        result = _squish(*cmd, timeout=120)
        assert result.returncode == 0

        msg = _extract_commit_message(result.stdout)
        if msg:
            assert len(msg) <= 72, f"Message too long ({len(msg)} chars): {msg!r}"

    def test_no_trailing_period(self, staged_repo, model_hint):
        """CLI should strip trailing periods from the generated message."""
        cmd = ["git", "--no-commit", str(staged_repo)]
        if model_hint:
            cmd = ["--model", model_hint, *cmd]

        result = _squish(*cmd, timeout=120)
        assert result.returncode == 0

        msg = _extract_commit_message(result.stdout)
        if msg:
            assert not msg.endswith("."), f"Message ends with period: {msg!r}"

    def test_no_staged_changes_exits_cleanly(self, tmp_path, model_hint):
        """squish git on a repo with no staged changes exits 0 with a helpful message."""
        repo = tmp_path / "clean"
        repo.mkdir()

        def git(*args):
            subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)

        git("init")
        git("config", "user.email", "t@t.com")
        git("config", "user.name",  "T")
        (repo / "x").write_text("x")
        git("add", "x")
        git("commit", "-m", "init")
        # Nothing staged now

        cmd = ["git", "--no-commit", str(repo)]
        if model_hint:
            cmd = ["--model", model_hint, *cmd]

        result = _squish(*cmd, timeout=30)
        assert result.returncode == 0, (
            f"Expected exit 0, got {result.returncode}\nstderr: {result.stderr}"
        )
        assert "staged" in result.stdout.lower(), (
            f"Expected 'staged' in output:\n{result.stdout}"
        )

    def test_jira_ticket_in_branch_name(self, tmp_path, model_hint):
        """When the branch name contains a Jira ticket, it should prefix the message."""
        repo = tmp_path / "jira_repo"
        repo.mkdir()

        def git(*args):
            subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)

        git("init")
        git("config", "user.email", "t@t.com")
        git("config", "user.name",  "T")
        (repo / "README.md").write_text("hello\n")
        git("add", "README.md")
        git("commit", "-m", "init")
        git("checkout", "-b", "feature/PROJ-1234-add-utils")

        (repo / "utils.py").write_text("def foo(): pass\n")
        git("add", "utils.py")

        cmd = ["git", "--no-commit", str(repo)]
        if model_hint:
            cmd = ["--model", model_hint, *cmd]

        result = _squish(*cmd, timeout=120)
        assert result.returncode == 0

        msg = _extract_commit_message(result.stdout)
        if msg:
            assert msg.startswith("PROJ-1234"), (
                f"Expected message to start with 'PROJ-1234', got: {msg!r}"
            )


# ── Test class: squish run pipe (shell-script path) ──────────────────────────

class TestSquishRunPipePath:
    """
    Tests the `squish run` pipe path used by git-commit-push-script.sh:

        echo "$PROMPT" | timeout 45 squish run [--model X] --max-tokens 60 --temperature 0.2

    This path is simpler (no ticket detection, no structured cleanup) but must
    respond within TIMEOUT_SECONDS and produce a usable first line.
    """

    # Matches what the shell script sends
    COMMIT_PROMPT = (
        "Git commit message (max 50 chars, no quotes/formatting):\n"
        "diff --git a/utils.py b/utils.py\n"
        "new file mode 100644\n"
        "index 0000000..abc1234\n"
        "--- /dev/null\n"
        "+++ b/utils.py\n"
        "@@ -0,0 +1,6 @@\n"
        "+def add(a, b):\n"
        "+    return a + b\n"
        "+\n"
        "+def subtract(a, b):\n"
        "+    return a - b\n"
    )

    def _run(self, model_hint, *, timeout=120):
        cmd = ["run", "--max-tokens", "60", "--temperature", "0.2"]
        if model_hint:
            cmd = ["--model", model_hint, *cmd]
        return _squish(*cmd, input_text=self.COMMIT_PROMPT, timeout=timeout)

    def test_pipe_returns_non_empty_response(self, model_hint):
        """Piping a commit prompt should produce a non-empty first line."""
        result = self._run(model_hint)

        assert result.returncode == 0, (
            f"squish run exited {result.returncode}\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )
        first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        assert first_line, "squish run produced no output for commit prompt"

    def test_pipe_first_line_length(self, model_hint):
        """
        The shell script takes only head -1, so the first line is the commit message.
        After cleanup it must fit within MAX_COMMIT_LENGTH (50 chars).
        Response may be longer before cleanup — just guard against total garbage.
        """
        result = self._run(model_hint)
        assert result.returncode == 0

        first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        # 200 chars is a sanity bound — the script will truncate further
        assert len(first_line) < 200, f"First line suspiciously long: {first_line!r}"

    def test_pipe_response_time(self, model_hint):
        """
        Once the server is warm, squish run should respond well within
        TIMEOUT_SECONDS (45s).  The previous tests warm the server, so this
        measures inference latency only.
        """
        t0 = time.perf_counter()
        result = self._run(model_hint)
        elapsed = time.perf_counter() - t0

        assert result.returncode == 0

        # 5s grace over the script's TIMEOUT_SECONDS for subprocess + HTTP overhead
        grace = 5
        limit = _SCRIPT_TIMEOUT_SECONDS + grace
        assert elapsed < limit, (
            f"squish run took {elapsed:.1f}s — exceeds "
            f"TIMEOUT_SECONDS={_SCRIPT_TIMEOUT_SECONDS} + {grace}s grace"
        )

    def test_pipe_no_markdown_in_response(self, model_hint):
        """
        The shell script strips markdown artefacts; verify the model doesn't
        produce egregious markdown that would survive the cleanup.
        """
        result = self._run(model_hint)
        assert result.returncode == 0

        first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        assert "```" not in first_line, f"Response contains code fence: {first_line!r}"
        # Hash-header at start is a common LLM mistake
        assert not first_line.startswith("#"), f"Response starts with #: {first_line!r}"
