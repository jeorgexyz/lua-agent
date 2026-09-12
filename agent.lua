-- agent.lua - The loop. This file is the artifact; everything else supports it.
--
-- One step is: build context -> call backend -> parse -> dispatch -> observe.
-- That is ten lines. Everything else in this file is the error taxonomy, and
-- the error taxonomy is the part that separates a demo from an agent.
--
-- Six terminal-ish states, each roughly ten lines, each with a test in
-- test.lua and a row in eval/ablate.lua:
--
--   parse_error  the model's output was not a tool call
--   tool_error   the tool raised, or the arguments failed validation
--   timeout      the tool ran too long
--   repeat       the same call came back N times; the agent is thrashing
--   denied       the approval gate refused
--   budget       the context window cannot fit even the next step
--
-- The unifying rule: A FAILURE IS AN OBSERVATION. Nothing here raises to the
-- caller. Every error is phrased for the model, appended to the transcript,
-- and the loop continues. That single decision is why recovery works at all,
-- and it is why the error strings in protocol.lua and tools.lua read like
-- instructions rather than diagnostics.

local context  = require('context')
local protocol = require('protocol')
local trace    = require('trace')

local Agent = {}
Agent.__index = Agent

-- opts:
--   backend   (required) table with :complete(prompt, opts) -> text, meta
--             The backend returns the full turn including the THOUGHT: and
--             CALL: prefixes, whatever scaffolding it used internally.
--   tools     (required) registry from tools.lua
--   tokenizer (required) anything with :encode(text) -> {ids}
--   max_steps        default 12
--   max_tokens       default 4096
--   repeat_threshold default 3
--   policy           context eviction policy name
--   approve          function(call) -> bool, for tools requiring approval
--   trace_path       JSONL destination; nil disables tracing
--   verbose          print the ledger and each step as it happens
--   fatal_on         set of statuses to treat as terminal instead of
--                    recoverable, e.g. {parse_error=true}. This exists so
--                    eval/ablate.lua can switch recovery OFF and measure
--                    what it was buying. Empty in normal operation -- the
--                    whole design rests on failures being observations.
function Agent.new(opts)
    opts = opts or {}
    if not opts.backend then error("Agent.new: needs a backend") end
    if not opts.tools then error("Agent.new: needs a tool registry") end
    if not opts.tokenizer then error("Agent.new: needs a tokenizer") end

    return setmetatable({
        backend = opts.backend,
        tools = opts.tools,
        approve = opts.approve,
        max_steps = opts.max_steps or 12,
        repeat_threshold = opts.repeat_threshold or 3,
        verbose = opts.verbose or false,
        fatal_on = opts.fatal_on or {},
        ctx = context.new(opts.tokenizer, opts.max_tokens or 4096, opts.policy),
        tracer = trace.open(opts.trace_path, opts.trace_path and "record" or "off"),
        steps = {},
    }, Agent)
end

-- A call's identity, for loop detection. Canonical JSON, so argument order
-- cannot disguise a repeat.
local function call_key(call)
    return trace.encode({ tool = call.tool, args = call.args or {} })
end

-- Real agents thrash by calling the same tool with the same arguments
-- forever. Ten lines stop it -- and without them, eval task `loop-bait`
-- runs to the step ceiling every time.
function Agent:is_repeating(call)
    local key, n = call_key(call), 0
    for _, s in ipairs(self.steps) do
        if s.call and call_key(s.call) == key then n = n + 1 end
    end
    return n >= self.repeat_threshold, n
end

-- Assemble the prompt for the next step and report what had to be evicted.
function Agent:build_prompt(task)
    local system = protocol.render_system(self.tools, task)
    local fixed = self.ctx:count(system)
    if fixed >= self.ctx.budget then
        return nil, nil, "the tool catalogue alone exceeds the context budget"
    end

    local kept, note = self.ctx:fit(fixed, self.steps)
    local body = protocol.render_transcript(kept)
    local parts = { system, "" }
    if note then parts[#parts + 1] = note end
    if body ~= "" then parts[#parts + 1] = body end
    local ledger = self.ctx:ledger(system, kept)
    ledger.evicted = note ~= nil

    -- The flat string is the protocol; the pieces are an optional courtesy.
    -- A chat-shaped backend (Ollama, any provider's messages API) can render
    -- the transcript as alternating turns, which is what an instruct model
    -- is trained on -- qwen2.5:0.5b re-called a tool it had already run when
    -- the whole history arrived as one blob. Backends that do not care keep
    -- using the string and nothing changes for them.
    return table.concat(parts, "\n"), ledger, nil, {
        system = system, steps = kept, note = note,
    }
end

-- One iteration. Split out from :run so tests can drive single steps and the
-- replay backend can be stepped deterministically.
--
-- Returns a step record: { thought, call, observation, status, tokens, ms }
function Agent:step(task)
    local prompt, ledger, budget_err, structured = self:build_prompt(task)
    if budget_err then
        return { status = "budget", observation = "Error: " .. budget_err }
    end

    local started = os.clock()
    local ok, raw, meta = pcall(self.backend.complete, self.backend, prompt, {
        stop = protocol.STOP,
        system = structured and structured.system,
        steps = structured and structured.steps,
        note = structured and structured.note,
        task = task,
        -- The catalogue + task, without the transcript: the opening user
        -- turn for a chat-shaped backend.
        task_block = structured and structured.system,
    })
    local elapsed = (os.clock() - started) * 1000

    -- A backend failure is the one thing that is genuinely fatal: there is no
    -- observation to feed back if the model itself is unreachable.
    if not ok then
        return { status = "fatal", observation = "Backend error: " .. tostring(raw) }
    end

    local rec = {
        prompt = prompt,
        raw = raw,
        ms = elapsed,
        evicted = ledger and ledger.evicted or nil,
        tokens = {
            window = ledger and ledger.total or 0,
            budget = ledger and ledger.budget or 0,
        },
    }
    if type(meta) == "table" then
        rec.tokens.completion = meta.tokens
        rec.legal_min = meta.min_legal
        rec.constrained = meta.constrained
    end

    local turn, perr = protocol.parse(raw)
    if not turn then
        -- The parser's message is written for the model; hand it straight
        -- back. This is the whole of parse recovery.
        rec.status = "parse_error"
        rec.observation = perr
        return rec
    end

    rec.thought = turn.thought

    if turn.kind == "answer" then
        rec.status = "answer"
        rec.call = { tool = protocol.ANSWER_TOOL, args = { text = turn.text } }
        rec.answer = turn.text
        return rec
    end

    rec.call = { tool = turn.tool, args = turn.args }

    local repeating, seen = self:is_repeating(rec.call)
    if repeating then
        rec.status = "repeat"
        rec.observation = string.format(
            "You have already called %s with these exact arguments %d times and "
            .. "the result has not changed. Try a different tool or different "
            .. "arguments, or answer with what you have.", turn.tool, seen)
        return rec
    end

    local ran, result, status = self.tools:invoke(turn.tool, turn.args, self.approve)
    rec.observation = result
    rec.status = ran and "continue" or status
    return rec
end

-- Run one task to completion. Returns answer, reason, steps.
-- reason: "answer" | "max_steps" | "budget" | "fatal"
function Agent:run(task)
    self.steps = {}
    local answer, reason = nil, "max_steps"

    for i = 1, self.max_steps do
        local rec = self:step(task)
        rec.step = i
        self.steps[#self.steps + 1] = rec
        self.tracer:write(rec)

        if self.verbose then self:print_step(rec) end

        if rec.status == "answer" then
            answer, reason = rec.answer, "answer"
            break
        elseif rec.status == "fatal" or rec.status == "budget" then
            reason = rec.status
            break
        elseif self.fatal_on[rec.status] then
            -- Recovery deliberately disabled for this status (ablation only).
            reason = rec.status
            break
        end
    end

    self.tracer:close()
    return answer, reason, self.steps
end

function Agent:print_step(rec)
    io.write(string.format("\n[step %d] %s\n", rec.step or 0, rec.status))
    if rec.tokens and rec.tokens.window and rec.tokens.window > 0 then
        io.write(string.format("  window: %d / %d tokens\n",
            rec.tokens.window, rec.tokens.budget))
    end
    if rec.legal_min then
        io.write(string.format("  grammar: narrowed to %d legal tokens at its tightest\n",
            rec.legal_min))
    end
    if rec.thought then io.write("  thought: " .. rec.thought .. "\n") end
    if rec.call then
        io.write("  call:    " .. trace.encode(rec.call) .. "\n")
    end
    if rec.observation then
        local obs = rec.observation:gsub("\n", "\n           ")
        io.write("  result:  " .. obs .. "\n")
    end
    io.flush()
end

return Agent
