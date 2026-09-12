-- backend/replay.lua - Deterministic playback from a recorded trace.
--
-- Returns the recorded completion for each step in order. No network, no
-- model, no key, no 60MB checkpoint. The test suite and every transcript in
-- examples/ run on this.
--
-- Divergence is an error, not a warning. If the loop asks for step 4 with a
-- prompt that does not match what was recorded, the agent's behaviour has
-- changed since the recording -- which is exactly the thing a regression test
-- exists to catch. Silently returning the old completion would hide it.

local trace = require('trace')

local backend = {}

local Replay = {}
Replay.__index = Replay

-- opts: { path = "traces/foo.jsonl", strict = true }
function backend.new(opts)
    opts = opts or {}
    if not opts.path then error("backend/replay: needs a trace path") end
    local steps = trace.read(opts.path)
    if #steps == 0 then
        error("backend/replay: " .. opts.path .. " is empty", 0)
    end
    return setmetatable({
        path = opts.path,
        steps = steps,
        i = 0,
        strict = opts.strict ~= false,
    }, Replay)
end

function Replay:complete(prompt, opts)
    self.i = self.i + 1
    local rec = self.steps[self.i]
    if not rec then
        error(string.format(
            "replay: the run asked for step %d but %s holds only %d. "
            .. "The agent is taking more steps than when this was recorded.",
            self.i, self.path, #self.steps), 0)
    end

    if self.strict and rec.prompt and rec.prompt ~= prompt then
        -- Report the first difference rather than dumping two prompts: they
        -- are hundreds of tokens each and differ by a few characters.
        local at = 1
        while at <= #prompt and at <= #rec.prompt
              and prompt:sub(at, at) == rec.prompt:sub(at, at) do
            at = at + 1
        end
        error(string.format(
            "replay: step %d prompt diverged from %s at byte %d\n"
            .. "  recorded: ...%s\n"
            .. "  now:      ...%s",
            self.i, self.path, at,
            rec.prompt:sub(math.max(1, at - 20), at + 40):gsub("\n", "\\n"),
            prompt:sub(math.max(1, at - 20), at + 40):gsub("\n", "\\n")), 0)
    end

    return rec.raw, {
        tokens = rec.tokens and rec.tokens.completion,
        min_legal = rec.legal_min,
        constrained = rec.constrained,
        replayed = true,
    }
end

return backend
