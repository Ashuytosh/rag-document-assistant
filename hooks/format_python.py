"""PostToolUse hook: auto-format a Python file after Claude edits/writes it.

Runs `ruff format` and `ruff check --fix` on the changed file so code stays
clean without manual formatting. Fails silently if ruff isn't installed, so
it never blocks the workflow.
"""
import json
import pathlib
import subprocess
import sys


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return

    file_path = (data.get("tool_input", {}) or {}).get("file_path", "") or ""
    if not file_path.endswith(".py"):
        return

    path = pathlib.Path(file_path)
    if not path.exists():
        return

    for cmd in (["ruff", "format", str(path)], ["ruff", "check", "--fix", str(path)]):
        try:
            subprocess.run(cmd, capture_output=True, check=False)
        except FileNotFoundError:
            return  # ruff not installed yet — skip quietly


if __name__ == "__main__":
    main()