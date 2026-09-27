"""Tools that only act inside a sandbox."""

from dataclasses import dataclass, field

from fontaine.agents.sandbox import Sandbox


@dataclass
class Trajectory:
    task: str
    steps: list[dict[str, str]] = field(default_factory=list)

    def add(self, kind: str, content: str, name: str = "") -> None:
        self.steps.append({"kind": kind, "name": name, "content": content})


class FilesystemTool:
    name = "filesystem"

    def read(self, sandbox: Sandbox, relative: str) -> str:
        path = sandbox.resolve(relative)
        if not path.is_file():
            raise FileNotFoundError(f"sandbox file not found: {relative}")
        return path.read_text(encoding="utf-8")

    def write(self, sandbox: Sandbox, relative: str, text: str) -> str:
        path = sandbox.resolve(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return f"wrote {relative}"


class PythonTool:
    name = "python"

    def run(self, sandbox: Sandbox, code: str) -> str:
        result = sandbox.run([sys_executable(), "-I", "-c", code])
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"python exited {result.returncode}")
        return result.stdout


def sys_executable() -> str:
    import sys

    return sys.executable


def run_episode(policy, sandbox: Sandbox, task: str, max_steps: int = 8) -> Trajectory:
    """Apply a policy until it finishes or ``max_steps`` is reached.

    ``policy`` is ``(trajectory) -> {"tool": name, "arguments": {...}}`` or
    ``{"finish": true, "answer": ...}``. The filesystem and python tools are
    the only ones this loop executes.
    """
    trajectory = Trajectory(task=task)
    trajectory.add("task", task)
    files = FilesystemTool()
    python = PythonTool()
    for _ in range(max_steps):
        action = policy(trajectory)
        if action.get("finish"):
            trajectory.add("termination", str(action.get("answer", "")))
            return trajectory
        name = action.get("tool")
        arguments = action.get("arguments") or {}
        trajectory.add("action", str(arguments), name=str(name))
        if name == "filesystem" and arguments.get("op") == "write":
            observation = files.write(sandbox, arguments["path"], arguments.get("text", ""))
        elif name == "filesystem" and arguments.get("op") == "read":
            observation = files.read(sandbox, arguments["path"])
        elif name == "python":
            observation = python.run(sandbox, arguments["code"])
        else:
            raise PermissionError(
                f"tool {name!r} is not available. This agent can use filesystem and python."
            )
        trajectory.add("observation", observation, name=str(name))
    trajectory.add("termination", "max_steps")
    return trajectory
