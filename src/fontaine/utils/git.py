"""Git metadata for experiment provenance."""

import subprocess
from pathlib import Path


def current_git_commit(repo_dir: str | Path | None = None) -> str:
    """Return the short git commit hash of the surrounding repo, or 'unknown'.

    Never raises: provenance metadata must not break a run in non-git contexts
    (installed packages, CI tarballs, user machines without git).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"
