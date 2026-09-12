# Lua Agent

<p align="center">
  <img width="350px" src="./assets/agent-lua.png" alt="Lua Agent">
</p>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

The sequel to [lua-llama](https://github.com/jeorgexyz/lua-llama) — that repo is the model this one drives.

> Where [lua-llama](https://github.com/jeorgexyz/lua-llama) is a minimal
> *model*, this is a minimal *agent*. The transformer forward pass is a
> solved, well-worn problem. The loop around it is not.

The loop itself is 155 lines of Lua; with its wire protocol, 273. No
dependencies — no HTTP library, no JSON library, no C extensions. Same rules
as repo 1.

---

## The claim

**Tool calling is a decoding constraint.**

At each step a model emits logits over the whole vocabulary. Set every token
that would break your grammar to `-inf` before sampling, and the model becomes
structurally incapable of producing a malformed tool call. No fine-tuning, no
provider API, no framework.

You can prove it with a model far too small to know what JSON is:

```
  arm            parsed as a tool call     picked a usable tool
  -----------    ---------------------     --------------------
  constrained     30/30   (100.0%)             0/30   (  0.0%)
  free             0/30   (  0.0%)             0/30   (  0.0%)
```

Same 15M checkpoint, same prompts, same greedy sampler. The only difference is
40 lines of logit masking in [`grammar.lua`](grammar.lua). Reproduce it with
`lua54 eval/ablate.lua`; the full transcript is in
[`examples/constrained-vs-free.txt`](examples/constrained-vs-free.txt).

**The two columns are the whole point.** The first is the grammar's work; the
second is the model's. Here is what the constrained arm actually emits:

```json
{"tool":"answer","args":{"text":"The 'A'.  One day, I am going to the park to play with my friends."}}
```

Syntactically perfect. Semantically nonsense. **The grammar guarantees the
syntax; the model supplies the semantics** — and a TinyStories checkpoint puts
that boundary under a microscope in a way a frontier model never could, because
a frontier model gets both right and you learn nothing about which is which.

---

## See it run

Writing a file, reading it back, and confirming — with the approval gate in
front of the write:

<p align="center">
  <img src="./assets/demo-approval-gate.gif" alt="the agent writing a file, reading it back, and confirming" width="660">
</p>

Error recovery — a bad path, the tool's error becomes the observation, the
model corrects itself and finishes:

<p align="center">
  <img src="./assets/demo-recovery.gif" alt="the agent recovering from a bad file path" width="640">
</p>

And the loop detector firing on a tool that keeps returning the same
unhelpful result, so the model gives up honestly instead of spinning:
[`demo-loop-detector.gif`](./assets/demo-loop-detector.gif). The refusal path
— the gate says no and the agent reports that rather than claiming success —
is [`examples/denied-write.txt`](examples/denied-write.txt).

These are recordings of the `replay` backend, so they run in 0.1s and are
generated rather than performed:

```bash
python tools/make_demos.py
```

Each one is driven by the `replay:` command printed in the matching
transcript, so a GIF cannot show behaviour the transcript does not claim.

---

## Requirements

- Lua **5.3+** (tested with Lua 5.4)
- A [lua-llama](https://github.com/jeorgexyz/lua-llama) checkout beside this
  one, at `../lua_llama` (override with `--llama-path`)

From that checkout the local backend needs a **checkpoint** (`stories15M.bin`,
60MB); everything else needs only **`tokenizer.bin`, 433KB**. Exact token
counting is a property of the tokenizer, not the model, so `--replay` and
`--backend http` get real BPE counts in ~57ms without loading a checkpoint.
Without the tokenizer the loop still runs and says its counts are estimated.

Nothing else. The HTTP backend shells out to `curl` rather than pulling in an
HTTP library, and `trace.lua` carries its own JSON codec.

---

## Usage

```bash
lua54 main.lua "What is 4871 * 209?"
```

Swap the model without touching the loop:

```bash
lua54 main.lua "<task>" --backend http                       # needs an API key
lua54 main.lua "<task>" --replay traces/calc-single.jsonl    # no model at all
lua54 main.lua "<task>" --no-constrain                       # the 0% arm
```

Everything else:

```bash
lua54 test.lua              # 430 assertions, no model, no network
lua54 eval/run.lua --ablate # what breaks when each mechanism is removed
lua54 eval/ablate.lua       # the parse-rate measurement above
lua54 record.lua            # regenerate traces/ and examples/
lua54 verify_cache.lua      # KV-cache reuse invariant (needs a checkpoint)
lua54 verify_examples.lua   # every transcript reproduces from its own command
python tools/make_demos.py  # regenerate the README GIFs
```

---

## Backends

| Backend | What it shows | Speed | Tested against |
|---|---|---|---|
| `local` | the **mechanism** — every layer readable, down to the matmul | ~3-4 tok/s | real checkpoints |
| `ollama` | the **capability**, locally and free | tens of tok/s | recorded fixtures |
| `http` | the **capability**, via the Anthropic API | network-bound | recorded fixtures |
| `replay` | **determinism** — the tests and every transcript in `examples/` | instant | recorded traces |

### Running a model that can actually do the tasks

`backend/local.lua` runs a model small enough to read end to end, and that
model cannot choose a tool — the eval scores 0/8 on both `stories15M` and
`stories42M`. Ollama is the cheapest way to close that gap: local, free, no
account.

```bash
ollama pull qwen2.5:1.5b-instruct
lua54 main.lua "What is 4871 * 209?" --backend ollama --model qwen2.5:1.5b-instruct
lua54 eval/run.lua --backend ollama --model qwen2.5:1.5b-instruct
```

```
  task                     pass   steps   tools it chose
  ------------------------ ----   -----   ------------------------
  calc-single              ok     2       calc,answer
  read-then-calc           ok     3       read_file,calc,answer
  recover-parse-error      ok     2       calc,answer
  recover-missing-file     FAIL   8       read_file x8
  loop-bait                FAIL   2       read_file,answer
  oversized-observation    ok     2       read_file,answer
  approved-write           ok     3       write_file,read_file,answer
  denied-write             ok     2       write_file,answer

  6/8 passed
```

The two failures are worth reading: `recover-missing-file` makes
eight read attempts at paths that do not exist and never tries `list_dir`,
and `loop-bait` invents a secret the config does not contain. Tool
*selection* and multi-step chaining work at 1.5B; recovering from a dead end
and refusing to confabulate do not.

**`grammar.lua` is not in the loop for this backend.** Ollama owns its
decoder, so the constraint that makes a 15M model emit valid JSON is simply
absent — this relies on the model being competent enough to follow the
format, which is the capability the grammar substitutes for. A pass here is
evidence about the *loop*, not about constrained decoding, and `eval/run.lua`
prints which decoder is in play for exactly that reason.

`--ollama-format` hands the call schema to Ollama, which applies it via
llama.cpp's GBNF — the same idea as `grammar.lua`, one layer down. The
comparison is the interesting part: our grammar on 15M against theirs on
0.5B.

**A finding worth the whole exercise.** Ollama compiles the JSON schema to
GBNF, which emits object properties in *declaration order*. `trace.lua` sorts
keys alphabetically so traces diff cleanly — and that sort rewrote the turn
schema from `{thought, call}` to `{call, thought}`, forcing the model to
commit to a tool *before* writing its reasoning. qwen2.5 responded by
re-calling a tool it had already run, five times running, with the thought
"The multiplication has been performed correctly." It read exactly like a
model too small for the job; a 3x larger model behaved identically. Key order
is meaningless in JSON right up until the JSON is a grammar.

**Why not a bigger local checkpoint instead?** Because the gap is training
data, not capacity — 15M → 42M tripled the parameters and quadrupled the
context and moved tool choice 0% → 0%. And the only instruct model that
converts to llama2.c format is TinyLlama-1.1B-Chat, which at measured
pure-Lua throughput is ~48 minutes per agent step.

The `http` transport is injectable, so request construction and response
handling are exercised against recorded fixtures rather than a live endpoint.
That is deliberate: the suite runs in CI with no secrets, no network flakiness
and no per-run cost. The 23 cases cover the shapes that actually break API
clients — thinking blocks preceding the text, a refusal returned at HTTP 200
rather than as an error status, an error object, an HTML error page, a
transport failure, and the loop treating a dead backend as fatal instead of
inventing an answer.

Pure-Lua inference runs ~3-4 tok/s at 15M, so ~0.05 tok/s at 1B. The local
backend will never complete a hard task. That is not a bug being apologised
for — it is why the ablation above is legible at all.

One disclosure that matters for reading the numbers: on the local backend the
harness writes the scaffolding (`THOUGHT: `, then `CALL: `) and the model fills
the slots. So the model chooses *which* tool and *what* arguments, not whether
to call one. The free arm measures the same checkpoint producing the whole turn
itself.

---

## MCP

The agent speaks [Model Context Protocol](https://modelcontextprotocol.io).
Mount a server and its tools join the registry beside the built-ins --
indistinguishable to the loop from there: same validation, same approval
gate, same grammar, same "a failure is an observation" rule.

```bash
lua54 main.lua "Echo hello" --backend ollama   --mcp "npx -y @modelcontextprotocol/server-everything"
```

```
mcp:     mcp-servers/everything (2025-06-18) -- 13 tools
[step 2] continue
  call:    {"args":{"message":"hello from the agent"},"tool":"mcp_echo"}
  result:  Echo: hello from the agent
```

`mcp.lua` is ~230 lines: `initialize`, `notifications/initialized`,
`tools/list`, `tools/call`, over either transport. That is the whole surface
needed to use a real server.

### Two transports, and why there are two

```bash
# stdio: server launched per call
lua54 main.lua "Echo hello" --mcp "npx -y @modelcontextprotocol/server-everything"

# streamable HTTP: one long-lived server, one session
npx -y @modelcontextprotocol/server-everything streamableHttp   # port 3001
lua54 main.lua "Echo hello" --mcp-url http://localhost:3001/mcp
```

**Lua has no bidirectional pipes.** `io.popen` opens a stream for reading
*or* writing, never both, so a long-lived stdio session is out of reach
without a C extension. The stdio transport works around it: write the whole
request sequence to a file, run the server with it as stdin, read what it
writes before it exits on EOF. Every call re-spawns the server and
re-handshakes.

That is slow, and worse, **it is wrong for any server that holds state** —
a re-spawned server loses the session, the open handle, the cursor. Speed is
just the visible symptom.

Streamable HTTP fixes both, in pure Lua, because the server is long-lived on
its own and `curl` only ever needs request/response. The `Mcp-Session-Id`
header returned by `initialize` is echoed on every later request, so one
session spans the whole run. Measured against `server-everything`:

| | discovery | per call | handshakes |
|---|---|---|---|
| stdio | 3.4s | ~3.4s | one per call |
| http | 0.24s | **0.06s** | one, total |

Roughly **57x** on tool calls, and state survives. Use stdio for servers
that only speak it; prefer `--mcp-url` otherwise.

**Two things the real server taught that the spec reads past**, both now
pinned by tests:

- Responses come back **out of order**, with notifications interleaved.
  Against `server-everything`, `notifications/tools/list_changed` arrived
  *before* the `initialize` result, and a call for id 3 returned before id
  2. Match on `id`, never position.
- A failing tool returns a **successful** JSON-RPC result carrying
  `isError: true`, not a JSON-RPC error. Treating it as an error loses the
  message -- and the message is good ("expected string, received undefined
  at message"), exactly the kind of observation this loop feeds back.

Mounted tools require approval by default; remote code does not get the
unattended treatment read-only built-ins get. `--mcp-trust` opts out,
`--mcp-prefix` namespaces them, and a name collision with an existing tool
is refused rather than silently shadowing it.

---

## What's in the loop

The interesting parts of an agent are not the happy path:

- **Constrained decoding** (`grammar.lua`) — an NFA over *token ids*, not
  characters, because BPE tokens straddle the boundaries the grammar cares
  about (`{"` is one token). Branching across tool paths is why it is an NFA.
- **Real token accounting** (`context.lua`) — every agent tutorial estimates
  the window as `#chars/4`. There is a real BPE tokenizer next door, so this
  counts for real and prints the ledger each step.
- **Eviction policies** — `drop_oldest`, `elide_observations`, `summarize`,
  swappable. When the window is trimmed the model is *told*, rather than left
  to hallucinate around the gap.
- **An explicit error taxonomy** (`agent.lua`) — `parse_error`, `tool_error`,
  `timeout`, `repeat`, `denied`, `budget`. Six named states, ~10 lines each,
  one test and one eval task each. Not buried retry logic.
- **A permission gate** (`tools.lua`) — read-only tools run unattended;
  writes need approval, default deny.
- **Deterministic replay** (`trace.lua`) — every step is one JSONL line, and
  `backend/replay.lua` plays them back.

The unifying rule, and the reason recovery works at all: **a failure is an
observation.** Nothing in the loop raises to the caller. Every error is phrased
for the model, appended to the transcript, and the loop continues — which is
why the error strings in `protocol.lua` and `tools.lua` read like instructions
rather than diagnostics.

---

## Verification

Repo 1's falsifiable claim is *"output matches llama2.c"*. This repo's is the
eval set: tasks with checkable answers, where `check` also sees the step list —
so producing the right answer *without calling the tool* still fails.

```
  task                       pass   steps  window
  -------------------------- ----   -----  ------
  calc-single                ok     2      160
  read-then-calc             ok     3      235
  recover-parse-error        ok     3      214
  recover-missing-file       ok     3      253
  loop-bait                  ok     7      468
  oversized-observation      ok     4      327
  approved-write             ok     3      233
  denied-write               ok     2      205

  8/8 passed
```

Then each mechanism is removed in turn. Every feature in `agent.lua` earns its
place by a row here:

```
  ablation                   pass    what breaks
  -------------------------- -----   ----------------------------------------
  (baseline)                 8/8     nothing
  no loop detector           7/8     loop-bait spins to the step ceiling
  no parse recovery          7/8     a truncated JSON call kills the run
  no tool-error recovery     7/8     one bad path kills the run
  no context eviction        7/8     oversized observation overflows
  policy=drop_oldest         8/8     nothing
```

That last row is a null result, left in deliberately: this eval does not
discriminate between eviction policies, so the choice of `elide_observations`
as the default is currently unjustified. See
[`docs/design.md`](docs/design.md).

**The loop ablation uses a scripted model** — a fixed turn list per task. That
is the point: an ablation of the loop needs the model held constant, or you
cannot tell whether removing the loop detector changed the outcome or the model
just sampled differently. The scripts encode mistakes real models make
(truncated calls, wrong paths, unhelpful re-reads). The claim about a real
model's judgement lives in the parse-rate table above, and its answer for 15M
params is: it has none.

---

## Project Structure

```
lua-agent/
├── main.lua          # CLI entry point
├── agent.lua         # the loop -- 155 lines; everything else supports it
├── protocol.lua      # prompt format, tool-call parse/serialize
├── tools.lua         # registry, validation, approval gate, built-ins
├── context.lua       # real token budget + eviction policies
├── grammar.lua       # logit masking -- constrained decoding
├── trace.lua         # JSONL record/replay + a minimal JSON codec
├── test.lua          # the whole suite; offline, no checkpoint needed
├── record.lua        # regenerate traces/ and examples/
├── verify_cache.lua  # KV-cache reuse invariant (needs a checkpoint)
├── verify_examples.lua # every transcript reproduces from its own command
├── tools/            # GIF generation for the README demos
├── backend/
│   ├── local.lua     # drives lua-llama's forward pass
│   ├── http.lua      # Anthropic Messages API (via curl)
│   └── replay.lua    # deterministic, no network
├── eval/
│   ├── ablate.lua    # the constrained-vs-free measurement
│   ├── run.lua       # pass rate + loop ablations
│   ├── tasks/        # the eval set
│   └── fixtures/     # files the tasks read
└── examples/         # real transcripts, generated not written
```

## Further reading

[`docs/design.md`](docs/design.md) — why the repo exists, what it deliberately
omits (no planning module, no RAG, no multi-agent), the five things building it
changed, and the open questions.

## License

MIT
