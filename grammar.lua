-- grammar.lua - Constrained decoding by logit masking.
--
-- This is the headline of the repo.
--
-- Tool calling is not a capability that emerges from scale. It is a decoding
-- constraint. At each step the model produces logits over the whole
-- vocabulary; set every token that would break the grammar to -inf before
-- sampling, and the model becomes structurally incapable of emitting a
-- malformed call. A 15M TinyStories checkpoint -- which has never seen JSON
-- and cannot reason -- then emits 100% parseable tool calls.
--
-- The grammar guarantees the SYNTAX. The model supplies the SEMANTICS. That
-- boundary is the whole lesson, and a 15M model puts it in sharp relief:
-- every call parses, and the arguments are nonsense.
--
--   {"tool":"read_file","args":{"path":"the little girl"}}
--    ^ grammar's work                    ^ model's work
--
--
-- WHY THIS IS HARDER THAN IT LOOKS
--
-- The grammar is defined over characters; it has to be enforced over BPE
-- tokens, and tokens straddle the character boundaries the grammar cares
-- about. `{"` is a single token in Llama's vocabulary. A token is legal only
-- if EVERY character it contributes is legal from the current state -- so the
-- machine has to simulate consuming a whole string, not test one character.
--
-- The second complication is branching: `{"tool":"` is shared by every tool,
-- but after the name the argument keys differ. So the machine is an NFA, not
-- a DFA. It tracks a SET of live states, one per tool path still consistent
-- with what has been emitted. The set collapses to one as soon as the model
-- commits to a name, and stays tiny throughout.
--
--
-- STRUCTURE
--
-- One linear path per tool, built from two segment kinds:
--
--   lit(text)   consume exactly this string
--   str         consume string-body characters; the closing quote belongs to
--               the NEXT literal, so termination falls out of the NFA rather
--               than needing lookahead
--
-- For calc, the path is:
--
--   lit  {"tool":"calc","args":{"expr":"
--   str
--   lit  "}}
--
-- There is no number state and no enum state. Numbers are strings on the
-- wire (see protocol.lua), and the tool name is baked into each path's first
-- literal, which is what makes the branching an NFA instead of a special case.

local trace = require('trace')

local grammar = {}

-- A string argument may not run forever. Without a cap the model can hold the
-- machine in the str state indefinitely and never close the call.
grammar.MAX_STRING = 256

-- Characters a JSON string body may contain here. Excluding the backslash as
-- well as the quote means no escape handling: the value is always literal,
-- which keeps both this machine and protocol.parse simple.
local function is_body_char(c)
    local b = c:byte()
    return b >= 0x20 and c ~= '"' and c ~= '\\'
end

--------------------------------------------------------------------------
-- Compilation
--------------------------------------------------------------------------

-- Precompute the decoded string for every vocabulary entry. Done once: the
-- mask runs on every generated token and cannot afford a tokenizer call.
--
-- Ids 0-2 (unk/BOS/EOS) decode to literal text like "<s>" in the llama2.c
-- tokenizer, which must never be treated as grammar input. They are blanked
-- here and handled explicitly in mask().
function grammar.build_token_table(tokenizer, vocab_size)
    local t = {}
    for id = 0, vocab_size - 1 do
        t[id] = (id <= 2) and "" or (tokenizer:decode(id) or "")
    end
    return t
end

local function path_for(tool)
    local segs = {}
    local head = string.format('{"tool":"%s","args":{', tool.name)
    local args = tool.args or {}

    if #args == 0 then
        segs[#segs + 1] = { kind = "lit", text = head .. "}}" }
        return segs
    end

    -- The machine writes the keys itself. The model only ever fills values,
    -- which is why a model that has never seen JSON still produces valid
    -- JSON: the parts it could get wrong are not its to write.
    for i, a in ipairs(args) do
        local open = (i == 1) and (head .. '"' .. a.name .. '":"')
                              or ('","' .. a.name .. '":"')
        segs[#segs + 1] = { kind = "lit", text = open }
        -- min = 1: a required argument may not be empty.
        --
        -- Without this the grammar emits calls that are syntactically
        -- perfect and guaranteed to fail validation, because tools.lua
        -- treats an empty required argument as a missing one. The two
        -- modules disagreed about what "well-formed" means and the model
        -- sat in the gap: stories42M emitted
        -- {"tool":"write_file","args":{"path":"","text":""}} four times
        -- running on the denied-write eval task, each rejected by the
        -- validator before the approval gate was ever consulted.
        --
        -- Enforcing it here rather than validating afterwards is the whole
        -- premise of this file: a constraint checkable during decoding
        -- should be unrepresentable, not caught.
        segs[#segs + 1] = { kind = "str", min = 1 }
    end
    segs[#segs + 1] = { kind = "lit", text = '"}}' }
    return segs
end

local Grammar = {}
Grammar.__index = Grammar

-- Compile from the tool registry. Required args only: an optional argument
-- would make the path branch, and a model that cannot reason has no business
-- choosing whether to supply one.
function grammar.compile(tools, tokenizer, vocab_size)
    vocab_size = vocab_size or tokenizer.vocab_size
    if not vocab_size then error("grammar.compile: need a vocab_size") end

    local paths, names = {}, {}
    for _, tool in ipairs(tools:list()) do
        local required = {}
        for _, a in ipairs(tool.args or {}) do
            if a.required then required[#required + 1] = a end
        end
        paths[#paths + 1] = path_for({ name = tool.name, args = required })
        names[#names + 1] = tool.name
    end
    if #paths == 0 then error("grammar.compile: registry is empty") end

    return setmetatable({
        paths = paths,
        names = names,
        vocab_size = vocab_size,
        token_text = grammar.build_token_table(tokenizer, vocab_size),
        mask_cache = {},
    }, Grammar)
end

--------------------------------------------------------------------------
-- The machine
--------------------------------------------------------------------------

local Machine = {}
Machine.__index = Machine

-- A live state is { path, seg, pos } where pos counts characters consumed
-- within the current segment. seg > #path means that path has completed.
-- The cache key for a state set.
--
-- Inside a string argument the exact character offset is NOT part of the
-- key. The legal-token set in a str state depends on the offset only through
-- the MAX_STRING cap, so every position far enough from that cap has an
-- identical mask -- and collapsing them is the difference between a 32000-
-- token rescan per generated token and a table lookup.
--
-- The margin must exceed the longest token in the vocabulary, since `consume`
-- walks a whole token's characters and a token near the cap can straddle it.
-- Llama's longest pieces are well under 32 bytes.
--
-- This only coarsens the CACHE KEY. The machine's real positions stay exact,
-- so nothing about the accepted language changes.
--
-- Two positions may share a key only if their legal-token sets are identical.
-- That rules out any offset below the segment's `min`, where the closing
-- quote is still illegal -- bucketing pos=0 together with pos=5 under a
-- min of 1 would hand the model a mask that lets it close an empty value.
local STR_BUCKET_MARGIN = 32

local function state_key(g, states)
    local parts = {}
    for _, s in ipairs(states) do
        local pos = s.pos
        local seg = g.paths[s.path][s.seg]
        if seg and seg.kind == "str"
           and pos >= (seg.min or 0)
           and pos <= grammar.MAX_STRING - STR_BUCKET_MARGIN then
            pos = "s"
        end
        parts[#parts + 1] = s.path .. ":" .. s.seg .. ":" .. tostring(pos)
    end
    table.sort(parts)
    return table.concat(parts, "|")
end

-- Advance one state by one character, appending every resulting state to
-- `out`. A str state produces two successors on a quote: the body cannot
-- contain one, so the only reading is that the value ended.
local function step_state(g, s, c, out)
    local path = g.paths[s.path]
    local seg = path[s.seg]
    if not seg then return end  -- already complete; nothing follows

    if seg.kind == "lit" then
        if seg.text:sub(s.pos + 1, s.pos + 1) == c then
            local pos = s.pos + 1
            if pos == #seg.text then
                out[#out + 1] = { path = s.path, seg = s.seg + 1, pos = 0 }
            else
                out[#out + 1] = { path = s.path, seg = s.seg, pos = pos }
            end
        end
        return
    end

    -- str: stay in the body, or let the following literal claim this char
    if s.pos < grammar.MAX_STRING and is_body_char(c) then
        out[#out + 1] = { path = s.path, seg = s.seg, pos = s.pos + 1 }
    end
    -- The value may only end once it has met its minimum length, so a
    -- required argument cannot be closed empty.
    if s.pos >= (seg.min or 0) then
        local nxt = path[s.seg + 1]
        if nxt and nxt.kind == "lit" and nxt.text:sub(1, 1) == c then
            if #nxt.text == 1 then
                out[#out + 1] = { path = s.path, seg = s.seg + 2, pos = 0 }
            else
                out[#out + 1] = { path = s.path, seg = s.seg + 1, pos = 1 }
            end
        end
    end
end

-- Consume a whole token's text. Returns the resulting state set, or nil if
-- the token is illegal from `states`.
local function consume(g, states, text)
    local cur = states
    for i = 1, #text do
        local nxt = {}
        local c = text:sub(i, i)
        for _, s in ipairs(cur) do step_state(g, s, c, nxt) end
        if #nxt == 0 then return nil end
        -- Deduplicate: shared prefixes across tool paths would otherwise
        -- multiply the set on every character.
        local seen, uniq = {}, {}
        for _, s in ipairs(nxt) do
            local k = s.path .. ":" .. s.seg .. ":" .. s.pos
            if not seen[k] then seen[k] = true uniq[#uniq + 1] = s end
        end
        cur = uniq
    end
    return cur
end

-- Fresh machine positioned at the start of a call, live on every tool path.
function Grammar:start()
    local states = {}
    for i = 1, #self.paths do
        states[i] = { path = i, seg = 1, pos = 0 }
    end
    return setmetatable({ g = self, states = states, emitted = {} }, Machine)
end

function Machine:is_done()
    for _, s in ipairs(self.states) do
        if s.seg > #self.g.paths[s.path] then return true end
    end
    return false
end

-- The legal-token set for the current state, as a list of token ids.
-- Cached: the machine revisits the same literal positions on every call, so
-- after warmup this is a table lookup rather than a 32000-token scan.
function Machine:legal_ids(vocab_size)
    local g = self.g
    local key = state_key(g, self.states)
    local cached = g.mask_cache[key]
    if not cached then
        cached = {}
        for id = 0, vocab_size - 1 do
            local text = g.token_text[id]
            if text ~= "" and consume(g, self.states, text) then
                cached[#cached + 1] = id
            end
        end
        g.mask_cache[key] = cached
    end
    return cached
end

-- Mask logits in place: every token illegal from the current state becomes
-- -inf, and legal ones keep the value the model produced. Returns how many
-- are left legal.
--
-- Preserving the values matters: masking must REMOVE FROM the model's
-- distribution, never flatten it. Zeroing the survivors would make every
-- legal token equally likely and throw away the only judgement the model is
-- contributing.
--
-- The returned count is worth logging. While the machine is writing
-- {"tool":" it drops to 1 -- the grammar is dictating and the model
-- contributes nothing. Inside an argument value it jumps to thousands, which
-- is exactly where the model's judgement is the only thing operating.
-- Watching that number rise and fall is watching the syntax/semantics
-- boundary move in real time.
function Machine:apply(logits, vocab_size)
    vocab_size = vocab_size or self.g.vocab_size

    -- Once complete, EOS is the only continuation.
    if self:is_done() then
        local saved = logits[3]   -- token id 2, 1-indexed
        for i = 1, vocab_size do logits[i] = -math.huge end
        logits[3] = saved
        return 1
    end

    local cached = self:legal_ids(vocab_size)
    local keep = {}
    for _, id in ipairs(cached) do keep[id + 1] = logits[id + 1] end
    for i = 1, vocab_size do logits[i] = -math.huge end
    for i, v in pairs(keep) do logits[i] = v end
    return #cached
end

-- Advance the machine by an accepted token.
function Machine:advance(token_id)
    local text = self.g.token_text[token_id]
    if not text or text == "" then return end
    local nxt = consume(self.g, self.states, text)
    if not nxt then
        -- Only reachable if a caller sampled a token the mask excluded.
        error(string.format(
            "grammar: token %d (%q) is not legal here -- was the mask applied?",
            token_id, text), 0)
    end
    self.states = nxt
    self.emitted[#self.emitted + 1] = text
end

function Machine:text()
    return table.concat(self.emitted)
end

-- Force the machine to completion, returning the characters still owed.
-- Used when generation hits its token ceiling mid-call: finishing the call
-- costs a few characters and turns a wasted step into a usable one.
function Machine:finish()
    local best
    for _, s in ipairs(self.states) do
        if not best or s.path < best.path then best = s end
    end
    if not best then return "" end
    local path, out = self.g.paths[best.path], {}
    local seg, pos = best.seg, best.pos
    while seg <= #path do
        local sg = path[seg]
        if sg.kind == "lit" then
            out[#out + 1] = sg.text:sub(pos + 1)
        elseif sg.min and pos < sg.min then
            -- A forced finish must still satisfy the minimum length, or it
            -- produces exactly the empty-argument call the min exists to
            -- prevent. "?" is a deliberate tell: the value was cut short by
            -- the token ceiling, not chosen.
            out[#out + 1] = string.rep("?", sg.min - pos)
        end
        seg, pos = seg + 1, 0
    end
    return table.concat(out)
end

-- Convenience for the ablation: does this text parse as a tool call?
function grammar.parses(text)
    local v = trace.decode(text)
    return type(v) == "table" and type(v.tool) == "string"
end

return grammar
