#!/usr/bin/env python3
"""Regenerate the README demo GIFs.

    python tools/make_demos.py                 # uses `lua54` from PATH
    LUA="C:\\Program Files\\Lua\\lua54.exe" python tools/make_demos.py

Two kinds of demo, and the difference matters:

REPLAY  Driven by the `replay:` command recorded in the matching transcript.
        Deterministic, offline, instant. A GIF cannot show behaviour the
        transcript does not claim, and verify_examples.lua fails first if
        the two ever disagree.

LIVE    A real model, and for MCP a real server. These cannot be replayed
        offline: the replay backend supplies the MODEL's turns, but tools
        still execute for real, so an MCP transcript needs the server up.
        They are recordings of actual runs, regenerated only when their
        prerequisites are present, and skipped with a note otherwise.

Prerequisites for the live set:
    ollama pull qwen2.5:1.5b-instruct
    npx -y @modelcontextprotocol/server-everything streamableHttp
"""

import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LUA = os.environ.get("LUA", "lua54")
LUA_Q = LUA if " " not in LUA else '"%s"' % LUA

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MCP_URL = os.environ.get("MCP_URL", "http://localhost:3001/mcp")
MODEL = os.environ.get("DEMO_MODEL", "qwen2.5:1.5b-instruct")

# transcript -> output gif
REPLAY_DEMOS = [
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
                return LUA_Q + " " + cmd.split(" ", 1)[1]
    raise SystemExit("no replay: line in " + path)


def reachable(url, timeout=3):
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True   # a 4xx still proves something is listening
    except Exception:
        return False


def ollama_has(model):
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=3) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
        return any(n == model or n.startswith(model) for n in names)
    except Exception:
        return False


def live_demos():
    """Live demos, paired with why they are being skipped when they are."""
    out = []
    have_model = ollama_has(MODEL)

    if have_model:
        out.append((
            "demo-ollama.gif",
            '%s main.lua "What is 4871 * 209?" --backend ollama --model %s '
            '--tools calc --max-steps 4' % (LUA_Q, MODEL),
            None,
        ))
    else:
        out.append((None, None,
                    "ollama model %s not pulled -- skipping demo-ollama.gif" % MODEL))

    mcp_up = reachable(MCP_URL)
    if have_model and mcp_up:
        out.append((
            "demo-mcp.gif",
            '%s main.lua "Echo the message: hello from MCP" --backend ollama '
            '--model %s --tools calc --mcp-url %s --mcp-trust --max-steps 3'
            % (LUA_Q, MODEL, MCP_URL),
            None,
        ))
    else:
        out.append((None, None,
                    "MCP server not at %s -- skipping demo-mcp.gif "
                    "(npx -y @modelcontextprotocol/server-everything streamableHttp)"
                    % MCP_URL))
    return out


def render(cmd, out, extra=None):
    args = [sys.executable, os.path.join(HERE, "make_gif.py"),
            "--cmd", cmd, "--out", os.path.join("assets", out)]
    if extra:
        args += extra
    return subprocess.run(args, cwd=ROOT).returncode


def main():
    failed = 0

    for transcript, out in REPLAY_DEMOS:
        if render(replay_command(transcript), out) != 0:
            failed += 1

    for out, cmd, skip in live_demos():
        if skip:
            print("skip: " + skip)
            continue
        # Live runs have a real result worth reading; hold the last frame
        # a beat longer than a replay.
        if render(cmd, out, ["--hold", "3.5"]) != 0:
            failed += 1

    if failed:
        raise SystemExit("%d demo(s) failed" % failed)


if __name__ == "__main__":
    main()
