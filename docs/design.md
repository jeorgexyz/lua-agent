# Design

Why this repo exists, what it claims, and what it deliberately does not do.

---

## The gap

A ReAct loop over a provider API is about 150 lines, and there are a hundred
blog posts that write them. That is not a sequel to a repo which boots a
transformer from a binary blob with no dependencies.

What makes llama2.c-style work land is two properties:

1. **Self-contained** — no network, no key, no account. You clone it and it runs.
2. **Falsifiable** — there is a number at the end you can check.

An API wrapper has neither. The interesting part happens on someone else's
server, and "it seemed to work" is the only available verdict. So the design
constraint for this repo is: keep both properties while moving up a level of
abstraction, from the model to the loop.

---

## The resolution: repo 1 is the backend

The agent drives the pure-Lua model from lua-llama. Self-contained again.

The obvious objection is that a 15M TinyStories model cannot do tool calling.
That is correct, and it is the most useful fact in the repo.

Tool calling is widely treated as an emergent capability that appears somewhere
around the frontier. It is not. It is **constrained decoding**: mask the logits
each step so only grammar-legal tokens survive, and malformed output becomes
unrepresentable. Fine-tuning makes a model *good* at choosing which tool; it is
not what makes the output parse.

A 15M model separates those two things cleanly, which no frontier model can do:

| | guaranteed by | evidence |
|---|---|---|
| **Syntax** — the call parses | the grammar | 100% parse rate at 15M |
| **Semantics** — the call is *right* | the model | ~0% task success at 15M |

Running the same ablation on a frontier model yields 100%/100% and teaches
nothing, because you cannot see which mechanism is responsible for what.

### Why this is honest rather than embarrassing

The local backend will never solve a task. Stated plainly and measured, that is
the result. Hidden behind a cherry-picked transcript, it would be a
misrepresentation. The README leads with the limitation for that reason.

---

## Three backends, one loop

| Backend | Demonstrates | Honest limitation |
|---|---|---|
| `local` | the mechanism, readable to the matmul | ~3-4 tok/s; never finishes a real task |
| `http` | the capability | needs a key; the model is a black box |
| `replay` | determinism, testability | replays only what was recorded |

The fact that these are interchangeable behind one `:complete(prompt, opts)`
method is itself the argument that the loop — not the model — is the artifact.

### Why the HTTP backend uses the text protocol, not native tool use

Every provider has a native tool-use API that is more reliable than parsing
text, and in production you should use it. This repo uses `protocol.lua` for
both backends anyway, because native tool use hides the exact mechanism the
repo exists to show, and because sharing one protocol is what makes the
backends swappable.

If you are building something real: use the native API. Then come back and
read `grammar.lua` to understand what it is doing for you.

---

## Where the difficulty actually is

The loop is easy. These are the parts that took the time:

**Token-level grammars.** The grammar is defined over characters but enforced
over BPE tokens, and tokens straddle character boundaries — `{"to` can be one
token. A token is legal only if *every* character it contributes is legal from
the current state. This requires precomputing the decoded string for all 32000
vocabulary entries at compile time; the mask runs once per generated token and
cannot afford a tokenizer call.

**KV cache reuse across steps.** The agent re-prompts with a growing transcript
every step. Refilling from scratch each time is O(n²) over a run and dominates
wall clock at 3 tok/s. Keeping the cache and forwarding only the delta is the
fix — and the invariant that catches bugs in it is the same one repo 1 used for
the speculative draft path: a warm run and a cold run must be byte-identical at
temperature 0.

Two optimizations lean on that invariant, and neither is obviously safe by
inspection. `feed` rewinds `pos` on a prefix divergence rather than clearing
the cache beyond it, and `reset` allocates once and afterwards only zeroes
`pos` — 8.4M Lua numbers per task saved on the 42M checkpoint. Both are
correct only if stale entries are unreachable, and if they are not, output
degrades silently instead of crashing. `verify_cache.lua` checks it against a
real checkpoint: warm, interleaved-order, and cold runs must agree exactly.
They do. This is the kind of claim that has to be executed rather than
reasoned about — the failure mode is invisible.

**Telling the model it lost information.** Silent truncation is the
single most common context bug. When eviction fires, a note goes into the
transcript saying what was dropped. A model that knows it forgot something asks
again; a model that does not, invents.

---

## Deliberate omissions

- **No planning module.** "Planning" in minimal agents is usually decoration. A
  `plan()` function that emits a numbered list and is never consulted again
  teaches nothing. What is real is a *comparison*: ReAct vs plan-and-execute on
  the same eval set, same trace format, printed side by side. If that
  comparison produces a difference worth showing, it earns a module. If it does
  not, that null result is worth publishing too.
- **No memory / RAG.** Out of scope; it is a retrieval problem wearing an agent
  costume.
- **No multi-agent.** Same loop, N times, plus a message bus. Nothing about the
  loop becomes clearer.
- **No streaming.** It would complicate the grammar machine for no pedagogical
  gain.
- **No async tool execution.** Lua has coroutines and it would be tempting. It
  would also double the complexity of `agent.lua`, which is the file that must
  stay readable.

---

## What building it changed

Six things the design did not anticipate, all measured rather than guessed.

**The grammar and the validator disagreed about "well-formed", and the model
sat in the gap.** `str` segments could consume zero characters, so the grammar
happily emitted `{"tool":"write_file","args":{"path":"","text":""}}` — perfect
JSON. But `tools.lua` treats an empty required argument as a missing one, so
`invoke` rejected it during validation and **never reached the approval gate**.
stories42M produced exactly this four times running on the `denied-write` eval
task. The eval reported "approval gate never fired", which was true and
explained nothing.

This is the repo's own thesis failing to hold inside the repo: a constraint
that is checkable during decoding should be *unrepresentable*, not caught
afterwards. Required values now carry `min = 1`, which needed three
coordinated changes — the transition rule, `finish()` (which force-completes a
call and would otherwise emit the very thing the minimum forbids), and the
mask cache key, where bucketing `pos=0` together with `pos=5` would have handed
the model a mask permitting an empty close. The last one is the interesting
failure: the optimization and the new constraint interact, and getting it wrong
would have silently re-opened the hole in exactly the states the cache serves.

It surfaced only because the eval prints which tools the model actually chose.
Before that column existed the row just said `FAIL`.

**stories15M's context is 256 tokens, and the tool catalogue did not fit.**
The verbose catalogue costs 194 tokens for two tools and 262 for four — so
with the default tool set the system prompt alone overflowed the model's
entire window before the task was even stated. `protocol.render_compact` drops
the format tutorial (the grammar enforces the format anyway, so explaining it
to a constrained model buys nothing) and renders one line per tool: 64 and 96
tokens. This is the most useful accident of driving a tiny model — context
budgeting stops being something you handle at 100k tokens and becomes the
first thing that breaks.

**Eviction cannot save you from one oversized observation.** Every policy here
evicts whole steps or whole observations, and the newest step can be neither
elided nor dropped — it is what the next turn responds to. A single tool
result larger than the budget therefore defeats all of them: `fit` returned a
window three times the budget while the ledger cheerfully reported the number.
The backstop is a last-resort truncation of the newest observation; the real
fix is capping output at the tool (`tools.MAX_OBSERVATION`). There is a test
pinning the invariant across all three policies now.

**The harness writes the scaffolding; the model fills the slots.** On the local
backend the prompt ends with `THOUGHT: `, the model free-generates one line,
the harness appends `CALL: `, and only then does the grammar engage. So the
model is not deciding *whether* to call a tool, only which one and with what
arguments. This is worth saying in the open because it changes how the headline
number should be read. The free arm measures what the same checkpoint does when
left to produce the whole turn itself.

**`answer` had to become a tool.** With the grammar covering every turn, a
separate ANSWER verb would live outside it, and a constrained model could then
never terminate. Making termination a tool call — `{"tool":"answer",...}` —
gave the loop one exit and the grammar one shape. Several real agent frameworks
land on the same design; it is clearer why after hitting the wall.

**One optimization that did not work, kept anyway.** Inside a string argument
the machine's state changes every token, so the mask cache missed and rescanned
all 32000 tokens per generated token. Bucketing mid-string positions into a
single cache key fixed the miss — and changed the wall clock not at all: 48s
per trial before, 48s after. Inference dominates completely at 3-4 tok/s. The
change stayed in (it avoids a 32000-entry allocation per token) but it bought
no time, and the comment in `grammar.lua` says so.

---

## Open questions

- Does the parse-rate result hold at temperature > 0, or does the grammar only
  look impressive under greedy decoding? Needs a sweep, same shape as repo 1's
  `temperature-sweep.txt`.
- Is `elide_observations` actually the right default? The current eval does not
  discriminate: `policy=drop_oldest` scores identically. Either the eval needs a
  task that separates them or the default is arbitrary — and saying so is
  better than implying a comparison that was never run.
- ~~Can the 42M checkpoint clear even one eval task?~~ **Answered: no, 0/7.**
  Its context is 1024 tokens rather than 256, so the window pressure that
  shapes everything on the 15M is simply absent — and it changes nothing.
  Capacity was never the binding constraint; judgement was. On six of seven
  tasks it answers immediately without touching a tool. The
  "local backend never finishes a task" claim stands as written.
- The loop ablation runs on a scripted model. A real-model ablation would be
  more convincing and much noisier; the honest version needs many seeds per
  cell, which at 3-4 tok/s is days of CPU.
