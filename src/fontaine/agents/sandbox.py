"""Permission boundaries for tool execution.

The sandbox roots every relative path and refuses to leave that directory.
Shell commands are an allowlist. Memory and CPU rlimits are applied on POSIX.
On Windows those two limits are refused instead of being ignored.
"""

import os
import subprocess
import sys
from pathlib import Path


class Sandbox:
    def __init__(
        self,
        root: Path,
        timeout_s: float = 5.0,
        allow_network: bool = False,
        memory_bytes: int | None = None,
        cpu_seconds: int | None = None,
        allow_commands: tuple[str, ...] = ("python", "python3"),
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.allow_network = allow_network
        self.memory_bytes = memory_bytes
        self.cpu_seconds = cpu_seconds
        self.allow_commands = allow_commands
        if sys.platform == "win32" and (memory_bytes or cpu_seconds):
            raise RuntimeError(
                "memory_bytes and cpu_seconds are enforced with POSIX rlimits. "
                "This process is Windows, so those limits cannot be applied. "
                "Omit them, or run the sandbox on Linux."
            )

    def resolve(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise PermissionError(
                f"path {relative!r} escapes the sandbox root {self.root}. "
                "Tools can only read and write inside that directory."
            )
        return candidate

    def run(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        if not argv:
            raise PermissionError("refusing to run an empty command")
        command = Path(argv[0]).stem.lower()
        if command not in self.allow_commands:
            raise PermissionError(
                f"command {command!r} is not in the sandbox allowlist {self.allow_commands}. "
                "Pass it in allow_commands if this process should be able to run it."
            )
        if not self.allow_network and command not in ("python", "python3"):
            raise PermissionError(
                f"network policy is deny, and {command!r} is not a confined interpreter. "
                "Set allow_network=true to run other allowlisted commands."
            )
        env = {"PATH": os.environ.get("PATH", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
        kwargs = {}
        if self.memory_bytes or self.cpu_seconds:
            kwargs["preexec_fn"] = self._limits
        return subprocess.run(
            argv,
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
            **kwargs,
        )

    def _limits(self) -> None:
        import resource

        if self.cpu_seconds:
            resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_seconds, self.cpu_seconds))
        if self.memory_bytes:
            resource.setrlimit(resource.RLIMIT_AS, (self.memory_bytes, self.memory_bytes))
