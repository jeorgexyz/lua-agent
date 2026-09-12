#!/usr/bin/env python3
"""Render a terminal session as an animated GIF.

    python tools/make_gif.py --cmd "lua54 main.lua ..." --out assets/demo.gif

Captures a command's output, then draws it line by line with a blinking
cursor. Used for the README demos.

Why a generator rather than a screen recorder: the demos run on the replay
backend, which is deterministic and finishes in ~0.1s. Regenerating them is
`make gifs`, not "set up OBS and perform the terminal correctly on camera".
The recordings therefore cannot drift from what the code actually prints --
if the output changes, the next regeneration shows it.
"""

import argparse
import os
import shutil
import subprocess
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise SystemExit(
        "make_gif needs Pillow:  pip install pillow\n"
        "It is the repo's only dependency, and only for regenerating the\n"
        "README GIFs -- nothing the agent runs on needs it.")

# A muted dark palette. Enough colour to read the structure, not a rainbow.
BG = (13, 17, 23)
FG = (201, 209, 217)
DIM = (110, 118, 129)
GREEN = (63, 185, 80)
YELLOW = (210, 153, 34)
BLUE = (88, 166, 255)
RED = (248, 81, 73)
CURSOR = (88, 166, 255)

PAD = 16
LINE_H = 20
FONT_SIZE = 15


def find_font():
    candidates = [
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\CascadiaMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, FONT_SIZE)
    return ImageFont.load_default()


def colour_for(line):
    """Colour by line role. Keyed to what main.lua actually prints."""
    stripped = line.strip()
    if stripped.startswith("[step"):
        if "parse_error" in line or "tool_error" in line or "fatal" in line:
            return RED
        if "repeat" in line or "denied" in line:
            return YELLOW
        if "answer" in line:
            return GREEN
        return BLUE
    if stripped.startswith("ANSWER:"):
        return GREEN
    if stripped.startswith(("thought:", "window:", "grammar:")):
        return DIM
    if stripped.startswith("call:"):
        return BLUE
    if stripped.startswith("result:") and "Error" in line:
        return RED
    if stripped.startswith(("task:", "backend:", "tools:")):
        return DIM
    if set(stripped) and set(stripped) <= set("-="):
        return DIM
    return FG


def render(lines, out_path, fps, hold_end, width_chars):
    font = find_font()
    # Measure with a representative glyph run; monospace so one is enough.
    bbox = font.getbbox("M" * 10)
    char_w = (bbox[2] - bbox[0]) / 10.0

    cols = max(width_chars, max((len(l) for l in lines), default=0))
    W = int(PAD * 2 + char_w * cols)
    H = PAD * 2 + LINE_H * (len(lines) + 1)

    frames = []
    for shown in range(len(lines) + 1):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        for i in range(shown):
            d.text((PAD, PAD + i * LINE_H), lines[i],
                   font=font, fill=colour_for(lines[i]))
        # Cursor on the line currently being written.
        cy = PAD + shown * LINE_H
        if shown < len(lines):
            d.rectangle([PAD, cy + 2, PAD + char_w, cy + LINE_H - 3], fill=CURSOR)
        frames.append(img)

    # Hold the finished frame so the result is readable before it loops.
    final = frames[-1]
    frames.extend([final] * max(1, int(hold_end * fps)))

    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=int(1000 / fps), loop=0, optimize=True,
    )
    return W, H, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", required=True, help="command to capture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--hold", type=float, default=2.5, help="seconds on the last frame")
    ap.add_argument("--width", type=int, default=76, help="minimum columns")
    ap.add_argument("--wrap", type=int, default=92,
                    help="hard-wrap output at this many columns")
    ap.add_argument("--cwd", default=".")
    args = ap.parse_args()

    # Note: do NOT re-wrap a quoted command here the way verify_examples.lua
    # has to. Python's subprocess already quotes correctly for cmd.exe;
    # adding another layer produces '""C:\Program' is not recognized.
    proc = subprocess.run(args.cmd, shell=True, cwd=args.cwd,
                          capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        sys.exit(f"command failed ({proc.returncode}): "
                 f"{(proc.stdout or '') + (proc.stderr or '')}"[:400])
    out = (proc.stdout or "") + (proc.stderr or "")
    raw_lines = [l.rstrip() for l in out.replace("\r\n", "\n").split("\n")]

    # Hard-wrap so one long tool result cannot blow the image out to 1500px.
    # Continuations are indented to the wrapped line's own indent so the
    # colour keying above still reads the right role.
    lines = []
    for line in raw_lines:
        if len(line) <= args.wrap:
            lines.append(line)
            continue
        indent = " " * (len(line) - len(line.lstrip()) + 2)
        while len(line) > args.wrap:
            cut = line.rfind(" ", 0, args.wrap)
            if cut <= len(indent):
                cut = args.wrap
            lines.append(line[:cut].rstrip())
            line = indent + line[cut:].lstrip()
        if line.strip():
            lines.append(line)

    while lines and not lines[-1]:
        lines.pop()
    if not lines:
        sys.exit("no output captured from: " + args.cmd)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    W, H, n = render(lines, args.out, args.fps, args.hold, args.width)
    size = os.path.getsize(args.out)

    # GIFs in a README should be small; flag it rather than silently shipping
    # a 5MB image.
    note = ""
    if size > 2_000_000:
        note = "  <- large for a README; consider fewer frames"
    print(f"{args.out}  {W}x{H}  {n} frames  {size/1024:.0f} KB{note}")


if __name__ == "__main__":
    main()
