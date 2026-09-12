-- backend/local.lua - Drives the pure-Lua model from repo 1.
--
-- The backend the README leads with: no API key, no network, the whole stack
-- readable from the agent loop down to the matmul.
--
-- It binds to lua-llama's exported forward pass rather than its generate()
-- loop, because the entire point is to intervene BETWEEN logits and
-- sampling -- that is where grammar.lua masks. The seam:
--
--   gen.forward(model, token, pos, state, key_cache, value_cache, att_buf)
--     -> logits[1..vocab_size]      (1-indexed array; token ids are 0-indexed)
--
-- No change to repo 1 is needed; forward() is already exported.
--
--
-- HOW A STEP IS GENERATED
--
-- The harness writes the scaffolding and the model fills the slots:
--
--   prompt ... "\nTHOUGHT: "     <- harness
--   "the sun was very big"       <- model, UNCONSTRAINED, one line
--   "\nCALL: "                   <- harness
--   {"tool":"answer","args":...  <- model, CONSTRAINED by grammar.lua
--
-- Stated plainly because it matters for reading the result: the model is not
-- deciding whether to call a tool. It is deciding WHICH tool and with what
-- arguments. A 15M checkpoint left to choose freely emits neither, which is
-- the 0% arm of the ablation and is measured in eval/ablate.lua rather than
-- assumed.
--
--
-- HONEST LIMITATION
--
-- Pure-Lua inference runs ~3-4 tok/s at 15M, so ~0.05 tok/s at 1B. This
-- backend demonstrates the MECHANISM. backend/http.lua demonstrates the
-- CAPABILITY. The split is the point, not an apology.

local grammar  = require('grammar')
local protocol = require('protocol')

local backend = {}

local Local = {}
Local.__index = Local

-- opts:
--   llama_path  path to the lua-llama checkout, default "../lua_llama"
--   checkpoint  default <llama_path>/stories15M.bin
--   tokenizer   default <llama_path>/tokenizer.bin
--   tools       registry, required when constrain is true
--   constrain   default true; false reproduces the 0% arm of the ablation
--   temperature default 0.0 (greedy, so runs are reproducible)
--   max_thought default 24 tokens
--   max_call    default 96 tokens
--   quiet       suppress repo 1's load-time prints
function backend.new(opts)
    opts = opts or {}
    local root = opts.llama_path or "../lua_llama"

    -- Repo 1's modules require() each other by bare name, so its directory
    -- has to be on the path as well as ours.
    package.path = root .. "/?.lua;" .. package.path

    local Model     = require('model')
    local Tokenizer = require('tokenizer')
    local gen       = require('generate')

    -- model.lua and tokenizer.lua print as they load. Useful in repo 1's CLI,
    -- noise here.
    local real_print = print
    if opts.quiet ~= false then _G.print = function() end end
    local ok, model = pcall(Model.new, opts.checkpoint or (root .. "/stories15M.bin"))
    local tok
    if ok then
        ok, tok = pcall(Tokenizer.new,
            opts.tokenizer or (root .. "/tokenizer.bin"), model.config.vocab_size)
    end
    _G.print = real_print
    if not ok then
        error("backend/local: " .. tostring(model or tok) ..
              "\n  looked in " .. root .. " -- set --llama-path if lua-llama is elsewhere", 0)
    end

    local self = setmetatable({
        model = model,
        tok = tok,
        gen = gen,
        cfg = model.config,
        constrain = opts.constrain ~= false,
        temperature = opts.temperature or 0.0,
        max_thought = opts.max_thought or 24,
        max_call = opts.max_call or 96,
        stats = { calls = 0, parsed = 0, tokens = 0 },
    }, Local)

    if self.constrain then
        if not opts.tools then error("backend/local: constrained mode needs the tool registry") end
        self.grammar = grammar.compile(opts.tools, tok, self.cfg.vocab_size)
    end

    self:reset()
    return self
end

-- Swap the tool set without reloading the model.
--
-- The grammar bakes the tool names and argument keys into its paths, so a
-- different tool set needs a different grammar -- but loading a checkpoint
-- takes ~17s and the eval runs one task per tool set. Recompiling is
-- milliseconds; reloading is not.
function Local:set_tools(reg)
    if self.constrain then
        self.grammar = grammar.compile(reg, self.tok, self.cfg.vocab_size)
    end
end

-- KV cache lifecycle.
--
-- The agent re-prompts with a growing transcript every step, so the naive
-- implementation refills the cache from scratch each time -- O(n^2) over a
-- run, and the dominant cost at 3 tok/s. Instead the cache persists and only
-- the tokens the prompt grew by are forwarded.
--
-- The invariant that catches bugs here is the one repo 1 used for the
-- speculative draft path: a warm run and a cold run must produce identical
-- output at temperature 0. eval/ablate.lua checks it.
function Local:reset()
    local c = self.cfg

    -- Allocate once. Re-zeroing on every reset is pure waste: the cache is
    -- position-indexed, and every slot below `pos` is written before it is
    -- ever read again -- the same invariant `feed` relies on when it rewinds
    -- after a prefix divergence. For the 42M checkpoint this is 8.4M Lua
    -- numbers per reset, and the eval calls reset once per task.
    if not self.key_cache then
        self.state = self.model:create_run_state()
        self.key_cache, self.value_cache = {}, {}
        for i = 1, c.n_layers * c.seq_len * c.kv_dim do
            self.key_cache[i], self.value_cache[i] = 0.0, 0.0
        end
        self.att_buf = {}
        for i = 1, c.seq_len do self.att_buf[i] = 0.0 end
    end

    self.pos = 0
    self.cached = {}   -- token ids currently represented in the cache
end

-- Forward `tokens` starting at the current position, reusing the cache for
-- any shared prefix. Returns the logits after the last token.
function Local:feed(tokens)
    local shared = 0
    while shared < #self.cached and shared < #tokens
          and self.cached[shared + 1] == tokens[shared + 1] do
        shared = shared + 1
    end

    -- Divergence means the cache past `shared` is stale. The KV cache is
    -- position-indexed, so rewinding is just moving pos back: entries beyond
    -- it are overwritten before they are ever read again.
    if shared < #self.cached then
        self.pos = shared
        for i = #self.cached, shared + 1, -1 do self.cached[i] = nil end
    end

    local logits
    for i = shared + 1, #tokens do
        if self.pos >= self.cfg.seq_len then
            error(string.format(
                "prompt exceeds the model's %d-token context", self.cfg.seq_len), 0)
        end
        logits = self.gen.forward(self.model, tokens[i], self.pos, self.state,
            self.key_cache, self.value_cache, self.att_buf)
        self.cached[i] = tokens[i]
        self.pos = self.pos + 1
    end
    return logits
end

local function sample(logits, n, temperature)
    if temperature < 0.01 then
        local best, best_v = 1, -math.huge
        for i = 1, n do
            if logits[i] > best_v then best, best_v = i, logits[i] end
        end
        return best - 1
    end
    -- Softmax over the masked distribution. -inf entries drop out naturally.
    local max_v = -math.huge
    for i = 1, n do if logits[i] > max_v then max_v = logits[i] end end
    local sum = 0
    local probs = {}
    for i = 1, n do
        probs[i] = math.exp((logits[i] - max_v) / temperature)
        sum = sum + probs[i]
    end
    local r, cdf = math.random() * sum, 0
    for i = 1, n do
        cdf = cdf + probs[i]
        if r < cdf then return i - 1 end
    end
    return n - 1
end

-- Generate freely until a stop string appears or the ceiling is hit.
function Local:generate_free(tokens, max_new, stops)
    local out, n = {}, self.cfg.vocab_size
    local logits = self:feed(tokens)
    for _ = 1, max_new do
        local id = sample(logits, n, self.temperature)
        if id == 2 then break end                      -- EOS
        local piece = self.tok:decode(id)
        out[#out + 1] = piece
        local text = table.concat(out)
        for _, s in ipairs(stops or {}) do
            if text:find(s, 1, true) then
                return text:sub(1, text:find(s, 1, true) - 1), #out
            end
        end
        tokens[#tokens + 1] = id
        logits = self:feed(tokens)
    end
    return table.concat(out), #out
end

-- Generate under the grammar. Every sampled token is legal by construction,
-- so the result parses by construction.
function Local:generate_constrained(tokens)
    local m = self.grammar:start()
    local n = self.cfg.vocab_size
    local min_legal = math.huge
    local produced = 0

    local logits = self:feed(tokens)
    for _ = 1, self.max_call do
        if m:is_done() then break end
        local legal = m:apply(logits, n)
        if legal < min_legal then min_legal = legal end
        local id = sample(logits, n, self.temperature)
        m:advance(id)
        produced = produced + 1
        tokens[#tokens + 1] = id
        if m:is_done() then break end
        logits = self:feed(tokens)
    end

    -- Hit the ceiling mid-call: finishing it costs a few characters and
    -- turns a wasted step into a usable one.
    local text = m:text()
    if not m:is_done() then text = text .. m:finish() end
    return text, produced, min_legal
end

function Local:complete(prompt, opts)
    opts = opts or {}

    -- The thought is free-form and one line long. It is not load-bearing for
    -- a 15M model -- it will be a fragment of a children's story -- but it is
    -- in the transcript, and seeing it next to the constrained call is what
    -- makes the syntax/semantics split legible.
    local thought_prompt = prompt .. "\n" .. protocol.THOUGHT_PREFIX
    local thought = self:generate_free(
        self.tok:encode(thought_prompt), self.max_thought, { "\n" })
    thought = thought:gsub("%s+$", ""):gsub("\n.*", "")

    local call_prompt = thought_prompt .. thought .. "\n" .. protocol.CALL_PREFIX
    local tokens = self.tok:encode(call_prompt)

    local call, produced, min_legal
    if self.constrain then
        call, produced, min_legal = self:generate_constrained(tokens)
    else
        call, produced = self:generate_free(tokens, self.max_call, { "\n" })
        call = call:gsub("^%s+", "")
    end

    self.stats.calls = self.stats.calls + 1
    self.stats.tokens = self.stats.tokens + produced
    if grammar.parses(call) then self.stats.parsed = self.stats.parsed + 1 end

    local text = protocol.THOUGHT_PREFIX .. thought .. "\n" .. protocol.CALL_PREFIX .. call
    return text, {
        tokens = produced,
        min_legal = min_legal ~= math.huge and min_legal or nil,
        constrained = self.constrain,
    }
end

return backend
