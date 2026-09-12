#!/usr/bin/env python3
"""Regenerate the README demo GIFs.

    python tools/make_demos.py                 # uses `lua54` from PATH
    LUA="C:\\Program Files\\Lua\\lua54.exe" python tools/make_demos.py

Each demo is driven by the `replay:` command that record.lua wrote into the
corresponding transcript, so a GIF cannot show behaviour the transcript does
not claim. If they ever disagree, verify_examples.lua fails first.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LUA = os.environ.get("LUA", "lua54")

# transcript -> output gif. Chosen to show the three mechanisms that are
# hard to believe without seeing them run.
DEMOS = [
    ("approved-write", "demo-approval-gate.gif"),
    ("recover-missing-file", "demo-recovery.gif"),
    ("loop-bait", "demo-loop-detector.gif"),
]


def replay_command(transcript):
    path = os.path.join(ROOT, "examples", transcript + ".txt")
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("replay:"):
                cmd = line.split(":", 1)[1].strip()
                # Substitute the interpreter by token, not by regex -- a
                # Windows path is full of backslashes and sed-style
                # replacement silently eats them (\L, \l, ...), which
                # produced "C:Program Filesuaua54.exe" and three identical
                # 2-line GIFs that looked plausible until opened.
                parts = cmd.split(" ", 1)
                return (LUA if " " not in LUA else '"%s"' % LUA) + " " + parts[1]
    raise SystemExit("no replay: line in " + path)


def main():
    failed = 0
    for transcript, out in DEMOS:
        cmd = replay_command(transcript)
        rc = subprocess.run(
            [sys.executable, os.path.join(HERE, "make_gif.py"),
             "--cmd", cmd, "--out", os.path.join("assets", out)],
            cwd=ROOT,
        ).returncode
        if rc != 0:
            failed += 1
    if failed:
        raise SystemExit("%d demo(s) failed" % failed)


if __name__ == "__main__":
    main()
