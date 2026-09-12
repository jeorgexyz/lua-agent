-- context.lua - Token budget and eviction.
--
-- Every minimal-agent tutorial estimates context as #chars/4 and calls it a
-- day. We have a real BPE tokenizer next door, so we count for real and print
-- the ledger. Knowing exactly what is in the window is most of context
-- management; the eviction policy is the small part.
--
-- The rule that matters more than the choice of policy: WHEN SOMETHING IS
-- DROPPED, THE MODEL IS TOLD. Silent truncation is the most common context
-- bug in real agents, and its signature is a model that confidently invents
-- the contents of an observation it can no longer see. A model that knows it
-- forgot something asks again.

local protocol = require('protocol')

local context = {}

local Context = {}
Context.__index = Context

-- tokenizer: anything with :encode(text) -> {ids}
-- budget:    hard ceiling in tokens
-- policy:    key of context.policies, default "elide_observations"
function context.new(tokenizer, budget, policy)
    if not tokenizer or not tokenizer.encode then
        error("context.new: needs a tokenizer with :encode")
    end
    local name = policy or "elide_observations"
    if not context.policies[name] then
        error("context.new: unknown policy " .. tostring(name))
    end
    return setmetatable({
        tokenizer = tokenizer,
        budget = budget or 4096,
        policy = name,
        cache = {},
        cache_n = 0,
    }, Context)
end

-- Price a string in real tokens. Memoized, because the system preamble and
-- tool catalogue would otherwise be re-tokenized on every single step -- and
-- BPE encoding is not cheap in pure Lua.
function Context:count(text)
    if text == nil or text == "" then return 0 end
    local hit = self.cache[text]
    if hit then return hit end
    local n = #self.tokenizer:encode(text)
    -- Bounded cache: observations are unique and would otherwise grow this
    -- without limit over a long run.
    if self.cache_n > 512 then self.cache, self.cache_n = {}, 0 end
    self.cache[text] = n
    self.cache_n = self.cache_n + 1
    return n
end

function Context:count_steps(steps)
    return self:count(protocol.render_transcript(steps))
end

-- The ledger: what is actually in the window right now.
-- main.lua prints this each step under --verbose, and it is the single most
-- useful thing to look at when an agent misbehaves.
function Context:ledger(system, steps)
    local sys = self:count(system)
    local hist = self:count_steps(steps)
    return {
        system = sys,
        transcript = hist,
        total = sys + hist,
        budget = self.budget,
        headroom = self.budget - (sys + hist),
        policy = self.policy,
    }
end

function Context:format_ledger(l)
    return string.format(
        "  context: system %d + transcript %d = %d / %d tokens (%d free, %s)",
        l.system, l.transcript, l.total, l.budget, l.headroom, l.policy)
end

-- Fit `steps` into the budget left after fixed costs.
-- Returns kept_steps, note. The note is appended to the transcript so the
-- model learns it lost information rather than hallucinating around the gap.
function Context:fit(fixed_tokens, steps)
    local available = self.budget - fixed_tokens
    local used = self:count_steps(steps)
    if used <= available then return steps, nil end

    local kept, note = context.policies[self.policy](steps, used - available, self)

    -- A policy can fail to free enough -- elision bottoms out once every
    -- observation is already a stub. Fall back to dropping whole steps rather
    -- than silently handing back an oversized window.
    if self:count_steps(kept) > available and self.policy ~= "drop_oldest" then
        local extra
        kept, extra = context.policies.drop_oldest(
            kept, self:count_steps(kept) - available, self)
        note = (note and (note .. " ") or "") .. (extra or "")
    end

    -- Last resort: a SINGLE observation can be larger than the entire
    -- budget, and no policy that evicts whole steps can help -- the newest
    -- step is the one being reasoned about and dropping it makes the next
    -- turn incoherent. So the observation itself is cut.
    --
    -- This is the floor of context management, and it is worth stating
    -- plainly: eviction cannot save you from one oversized tool result.
    -- Capping output at the tool (tools.MAX_OBSERVATION) is the real fix;
    -- this is the backstop that keeps the request sendable when it is not
    -- enough. Without it `fit` returns an over-budget window and the ledger
    -- reports a number the model never actually received.
    local over = self:count_steps(kept) - available
    if over > 0 and #kept > 0 then
        local last = kept[#kept]
        if last.observation and #last.observation > 0 then
            local copy = {}
            for k, v in pairs(last) do copy[k] = v end
            -- Cut by characters, then verify in tokens: the ratio varies.
            local keep_chars = #last.observation
            while keep_chars > 0 and self:count_steps(kept) - available > 0 do
                keep_chars = math.floor(keep_chars * 0.6)
                copy.observation = last.observation:sub(1, keep_chars)
                    .. string.format("\n... [cut: %d of %d bytes shown]",
                        keep_chars, #last.observation)
                copy.truncated = true
                kept[#kept] = copy
            end
            note = (note and (note .. " ") or "")
                .. "[context: the most recent tool result was too large for the "
                .. "window and was cut. Read a smaller portion if you need the rest.]"
        end
    end

    return kept, note
end

--------------------------------------------------------------------------
-- Policies. Each takes (steps, tokens_to_free, ctx) and returns
-- kept_steps, note. Swapping these and re-running the eval is the ablation.
--------------------------------------------------------------------------
context.policies = {}

-- Truncate old observations to a stub, keeping every thought and call.
-- The usual right default: tool output is bulky, and the model mostly needs
-- to remember THAT it called something and with what, not the full result.
function context.policies.elide_observations(steps, to_free, ctx)
    local kept, freed, n = {}, 0, 0
    for i, s in ipairs(steps) do
        kept[i] = s
    end
    -- Oldest first; the most recent observation is the one being reasoned
    -- about right now, so it is elided last and only under real pressure.
    for i = 1, #kept - 1 do
        if freed >= to_free then break end
        local s = kept[i]
        if s.observation and not s.elided then
            local before = ctx:count(s.observation)
            local stub = string.format("[%d bytes elided]", #s.observation)
            local after = ctx:count(stub)
            if before > after then
                local copy = {}
                for k, v in pairs(s) do copy[k] = v end
                copy.observation = stub
                copy.elided = true
                kept[i] = copy
                freed = freed + (before - after)
                n = n + 1
            end
        end
    end
    if n == 0 then return kept, nil end
    return kept, string.format(
        "[context: %d earlier tool result%s elided to stay within the window. "
        .. "Call the tool again if you need the full output.]",
        n, n == 1 and " was" or "s were")
end

-- Drop whole steps from the front, keeping the most recent work.
function context.policies.drop_oldest(steps, to_free, ctx)
    local freed, cut = 0, 0
    -- Never drop the final step: it holds the observation the next turn is a
    -- response to.
    while cut < #steps - 1 and freed < to_free do
        cut = cut + 1
        freed = freed + ctx:count(protocol.render_transcript({ steps[cut] }))
    end
    if cut == 0 then return steps, nil end
    local kept = {}
    for i = cut + 1, #steps do kept[#kept + 1] = steps[i] end
    return kept, string.format(
        "[context: %d earlier step%s dropped to stay within the window.]",
        cut, cut == 1 and " was" or "s were")
end

-- Replace the oldest run of steps with a model-written summary. Costs an
-- extra backend call, which is why it is not the default -- and why it needs
-- a summarizer wired in explicitly.
function context.policies.summarize(steps, to_free, ctx)
    if not ctx.summarizer then
        -- Degrade loudly rather than silently doing something else.
        local kept, note = context.policies.drop_oldest(steps, to_free, ctx)
        return kept, (note or "") .. " [no summarizer configured; dropped instead]"
    end
    local half = math.max(1, math.floor(#steps / 2))
    local old = {}
    for i = 1, half do old[i] = steps[i] end
    local ok, summary = pcall(ctx.summarizer, protocol.render_transcript(old))
    if not ok or type(summary) ~= "string" then
        return context.policies.drop_oldest(steps, to_free, ctx)
    end
    local kept = { { thought = nil, observation = nil, note =
        "[context: earlier steps summarized] " .. summary } }
    for i = half + 1, #steps do kept[#kept + 1] = steps[i] end
    return kept, string.format("[context: %d earlier steps were summarized.]", half)
end

return context
