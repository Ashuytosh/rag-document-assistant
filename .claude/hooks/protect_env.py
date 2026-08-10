"""PreToolUse hook: block any Read/Edit/Write that targets a .env file.

Claude Code passes the tool call as JSON on stdin. We inspect the target
file path and exit with code 2 to block the call if it is a .env file.
Exit 0 lets the call proceed.
"""

import json
import sys


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # if we can't parse, don't block

    file_path = (data.get("tool_input", {}) or {}).get("file_path", "") or ""
    name = file_path.replace("\\", "/").rstrip("/").split("/")[-1]

    if name == ".env" or name.startswith(".env."):
        print(
            "Blocked: .env files are protected. "
            "Check .env.example for the list of variable names instead.",
            file=sys.stderr,
        )
        sys.exit(2)  # exit code 2 blocks the tool call

    sys.exit(0)


if __name__ == "__main__":
    main()
