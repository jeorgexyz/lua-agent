-- test.lua - The whole test suite. Runs offline, no model, no key.
--
--   lua54 test.lua
--
-- Everything here must stay runnable by someone who just cloned the repo and
-- has no checkpoint. Tests that need a model live in eval/, not here.

package.path = "./?.lua;" .. package.path

local pass, fail = 0, 0
local current = "?"

local function group(name) current = name end

local function check(desc, ok, detail)
    if ok then
        pass = pass + 1
    else
        fail = fail + 1
        io.write(string.format("FAIL  %s: %s\n", current, desc))
        if detail then io.write("        " .. tostring(detail) .. "\n") end
    end
end

local function eq(desc, got, want)
    check(desc, got == want, string.format("got %s, want %s",
        tostring(got), tostring(want)))
end

--------------------------------------------------------------------------
group("trace/json")
--------------------------------------------------------------------------
local trace = require("trace")

do
    local cases = {
        { tool = "calc", args = { expr = "4871 * 209" } },
        { a = {}, b = { 1, 2, 3 }, d = true, e = 1.5, f = 42 },
        { s = 'quote " backslash \\ newline \n tab \t' },
        { nested = { deep = { deeper = { "x" } } } },
        { empty_args = {} },
    }
    for i, v in ipairs(cases) do
        local enc = trace.encode(v)
        local dec, err = trace.decode(enc)
        check("round-trip " .. i .. " decodes", dec ~= nil, err)
        if dec then
            eq("round-trip " .. i .. " is stable", trace.encode(dec), enc)
        end
    end

    -- Key order must be deterministic or traces stop diffing cleanly
    eq("keys sorted", trace.encode({ z = 1, a = 2, m = 3 }),
       '{"a":2,"m":3,"z":1}')
    eq("empty table is an object", trace.encode({}), "{}")
    eq("integers have no decimal point", trace.encode({ n = 42 }), '{"n":42}')

    -- Decode must reject rather than raise
    for _, bad in ipairs({ "{bad}", '{"a":1} trailing', '{"a":', '"unterminated' }) do
        local v, e = trace.decode(bad)
        check("rejects " .. bad, v == nil and type(e) == "string", e)
    end
end

--------------------------------------------------------------------------
group("protocol")
--------------------------------------------------------------------------
local ok_protocol, protocol = pcall(require, "protocol")
if ok_protocol and protocol.parse then
    local p = protocol.parse('THOUGHT: I should multiply.\nCALL: {"tool":"calc","args":{"expr":"2*3"}}')
    if check("parses a well-formed call", p ~= nil and p.kind == "call") then end
    if p and p.kind == "call" then
        eq("extracts tool name", p.tool, "calc")
        eq("extracts args", p.args and p.args.expr, "2*3")
        eq("extracts thought", p.thought, "I should multiply.")
    end

    local a = protocol.parse('THOUGHT: done\nCALL: {"tool":"answer","args":{"text":"6"}}')
    check("answer is a tool call", a ~= nil and a.kind == "answer", a and a.kind)
    if a and a.kind == "answer" then eq("answer text", a.text, "6") end

    -- Malformed input must return an error string, never raise: that string
    -- is fed back to the model as the observation.
    for _, bad in ipairs({
        "no call here at all",
        'CALL: {"tool":"calc"',
        'CALL: {"args":{}}',
        'CALL: {"tool":"calc","args":"not an object"}',
    }) do
        local v, e = protocol.parse(bad)
        check("rejects: " .. bad:sub(1, 34), v == nil and type(e) == "string", e)
    end
end

--------------------------------------------------------------------------
group("tools")
--------------------------------------------------------------------------
local ok_tools, tools = pcall(require, "tools")
if ok_tools and tools.registry then
    local reg = tools.registry()
    reg:add(tools.calc)

    eq("calc multiplies", select(2, reg:invoke("calc", { expr = "4871 * 209" })), "1018039")
    eq("calc adds", select(2, reg:invoke("calc", { expr = "2 + 2" })), "4")

    -- Validation failures are observations, not crashes
    local ok1, err1 = reg:validate("calc", {})
    check("missing required arg rejected", not ok1, err1)
    local ok2, err2 = reg:validate("nope", { expr = "1" })
    check("unknown tool rejected", not ok2, err2)

    -- calc must not be a Lua eval hole
    local _, res = reg:invoke("calc", { expr = "os.exit(1)" })
    check("calc refuses non-arithmetic", tostring(res):match("arithmetic") ~= nil, res)

    -- The approval gate must default to refusing
    reg:add(tools.write_file)
    local _, _, status = reg:invoke("write_file",
        { path = "x.txt", text = "y" }, function() return false end)
    eq("denied write reports denied", status, "denied")
end

--------------------------------------------------------------------------
group("context")
--------------------------------------------------------------------------
local ok_ctx, context = pcall(require, "context")
if ok_ctx and context.new then
    -- A stand-in tokenizer: one token per 4 chars. The real one is only
    -- needed to prove the counts are real, not to prove the policies work.
    local fake_tok = { encode = function(_, s)
        local t = {}
        for _ = 1, math.max(1, math.ceil(#s / 4)) do t[#t + 1] = 0 end
        return t
    end }

    local ctx = context.new(fake_tok, 100, "elide_observations")
    eq("counts via the tokenizer", ctx:count("aaaaaaaa"), 2)
    eq("count is memoized", ctx:count("aaaaaaaa"), 2)

    local steps = {}
    for i = 1, 10 do
        steps[i] = {
            thought = "t" .. i,
            call = { tool = "read_file", args = { path = "f" .. i } },
            observation = string.rep("x", 200),
        }
    end

    local kept, note = ctx:fit(10, steps)
    check("eviction fires under pressure", note ~= nil and note ~= "", note)
    check("eviction keeps recent steps", #kept > 0)
    check("result fits the budget",
        ctx:count(table.concat({ note or "" })) < 100)

    -- The model must be TOLD it lost information. Silent truncation is the
    -- single most common context bug.
    check("note names what was dropped",
        note and (note:match("elid") or note:match("drop") or note:match("omit")), note)

    -- fit() must never hand back a window bigger than the budget. The hard
    -- case is a SINGLE observation larger than the whole budget: no policy
    -- that evicts whole steps can help, so the observation itself gets cut.
    -- Before that backstop existed, this returned ~3x the budget and the
    -- ledger reported a number the model never received.
    local tight = context.new(fake_tok, 120, "elide_observations")
    local huge = { { thought = "t", call = { tool = "read_file", args = {} },
                     observation = string.rep("y", 8000) } }
    local kept2 = tight:fit(10, huge)
    check("one oversized observation still fits the budget",
        tight:count_steps(kept2) <= 110,
        "got " .. tight:count_steps(kept2) .. " tokens, budget 120 minus 10 fixed")

    -- And the same must hold across every policy.
    for _, name in ipairs({ "drop_oldest", "elide_observations", "summarize" }) do
        local c = context.new(fake_tok, 120, name)
        local k = c:fit(10, huge)
        check("policy " .. name .. " respects the budget",
            c:count_steps(k) <= 110, "got " .. c:count_steps(k))
    end
end

--------------------------------------------------------------------------
group("grammar")
--------------------------------------------------------------------------
local ok_gram, grammar = pcall(require, "grammar")
if ok_gram and grammar.compile then
    -- A tiny fake tokenizer with a known vocabulary lets us test the token
    -- machine without loading a 60MB checkpoint. Character-level plus a few
    -- multi-character tokens, which is the case that actually matters:
    -- BPE tokens straddle the character boundaries the grammar cares about.
    local pieces = {
        [0] = "<unk>", [1] = "<s>", [2] = "</s>",
    }
    local n = 3
    local function add(s) pieces[n] = s n = n + 1 return n - 1 end
    for c = 32, 126 do add(string.char(c)) end
    local multi = { '{"', 'tool', '":"', 'calc', 'args', '"}', '}}', '","',
                    'read_file', 'answer', 'text', 'expr', 'path', '2', '09', '4871' }
    for _, m in ipairs(multi) do add(m) end
    local vocab_size = n

    local fake_tok = {
        vocab_size = vocab_size,
        decode = function(_, id) return pieces[id] or "" end,
    }

    local reg = tools.registry()
    reg:add(tools.calc)
    reg:add(tools.read_file)

    local g = grammar.compile(reg, fake_tok, vocab_size)
    check("compiles", g ~= nil)

    if g then
        -- Drive the machine greedily and confirm the result parses. The mask
        -- is the only thing steering here; there is no model in this test.
        local m = g:start()
        local out, guard = {}, 0
        while not m:is_done() and guard < 400 do
            guard = guard + 1
            local logits = {}
            for i = 1, vocab_size do logits[i] = 0 end
            local legal = m:apply(logits, vocab_size)
            check("some token is always legal (step " .. guard .. ")", legal > 0)
            if legal == 0 then break end
            -- Take the lowest-id legal token: an adversarial "model" that
            -- knows nothing. If this still parses, the grammar is doing it.
            local chosen
            for i = 1, vocab_size do
                if logits[i] > -math.huge then chosen = i - 1 break end
            end
            out[#out + 1] = fake_tok:decode(chosen)
            m:advance(chosen)
        end
        check("machine terminates", m:is_done(), "after " .. guard .. " tokens")

        local text = table.concat(out)
        local decoded, derr = trace.decode(text)
        check("adversarial decode still parses: " .. text, decoded ~= nil, derr)

        -- A required argument must be non-empty BY CONSTRUCTION, not by
        -- later validation. tools.lua treats an empty required arg as a
        -- missing one, so a grammar that permits "" emits calls that are
        -- syntactically perfect and guaranteed to fail -- and the approval
        -- gate is never even reached, because validation rejects first.
        -- stories42M hit this four times running on the denied-write task.
        if decoded and decoded.args then
            for k, v in pairs(decoded.args) do
                check("argument " .. k .. " is non-empty", v ~= "",
                    "the grammar allowed an empty required value")
            end
        end
        if decoded then
            check("has a tool field", decoded.tool ~= nil)
            check("tool is a registered name",
                decoded.tool == "calc" or decoded.tool == "read_file"
                or decoded.tool == "answer", decoded.tool)
            check("has an args object", type(decoded.args) == "table")
        end

        -- Directly: from the start of a value, the closing quote must be
        -- illegal. This is the invariant the walk above only samples.
        local m2 = g:start()
        local prefix = '{"tool":"calc","args":{"expr":"'
        local by_char = {}
        for id = 0, vocab_size - 1 do
            local t = fake_tok:decode(id)
            if #t == 1 and by_char[t] == nil then by_char[t] = id end
        end
        local drove = true
        for i = 1, #prefix do
            local id = by_char[prefix:sub(i, i)]
            if id == nil then drove = false break end
            local okc = pcall(function() m2:advance(id) end)
            if not okc then drove = false break end
        end
        check("drove the machine to a value start", drove)
        if drove then
            local lg = {}
            for i = 1, vocab_size do lg[i] = 0 end
            m2:apply(lg, vocab_size)
            check("closing quote illegal at an empty required value",
                lg[by_char['"'] + 1] == -math.huge)
            -- and legal once one character is in
            m2:advance(by_char["7"])
            local lg2 = {}
            for i = 1, vocab_size do lg2[i] = 0 end
            m2:apply(lg2, vocab_size)
            check("closing quote legal after one character",
                lg2[by_char['"'] + 1] > -math.huge)
        end

        -- finish() force-completes a call; it must not produce the empty
        -- value the minimum exists to forbid.
        local m3 = g:start()
        for i = 1, #prefix do
            local id = by_char[prefix:sub(i, i)]
            if id then pcall(function() m3:advance(id) end) end
        end
        local forced = trace.decode(m3:text() .. m3:finish())
        check("forced finish still yields a parseable call", forced ~= nil)
        if forced and forced.args then
            for k, v in pairs(forced.args) do
                check("forced finish leaves " .. k .. " non-empty", v ~= "")
            end
        end
    end
end

--------------------------------------------------------------------------
group("agent")
--------------------------------------------------------------------------
local ok_agent, Agent = pcall(require, "agent")
if ok_agent and Agent.new and ok_tools then
    -- A scripted backend: the cheapest possible way to test the loop's
    -- control flow, including every arm of the error taxonomy.
    local function scripted(replies)
        local i = 0
        return { complete = function(_, _, _)
            i = i + 1
            return replies[i] or 'THOUGHT: giving up\nCALL: {"tool":"answer","args":{"text":"none"}}', {}
        end }
    end

    local reg = tools.registry()
    reg:add(tools.calc)

    do
        local a = Agent.new({
            backend = scripted({
                'THOUGHT: multiply\nCALL: {"tool":"calc","args":{"expr":"4871 * 209"}}',
                'THOUGHT: done\nCALL: {"tool":"answer","args":{"text":"1018039"}}',
            }),
            tools = reg, tokenizer = { encode = function(_, s) return { #s } end },
        })
        local answer, reason, steps = a:run("What is 4871 * 209?")
        eq("happy path answers", answer, "1018039")
        eq("happy path reason", reason, "answer")
        eq("happy path took two steps", #steps, 2)
        eq("observation fed back", steps[1].observation, "1018039")
    end

    do  -- parse error recovers rather than crashing
        local a = Agent.new({
            backend = scripted({
                'THOUGHT: oops\nCALL: {"tool":"calc"',
                'THOUGHT: retry\nCALL: {"tool":"calc","args":{"expr":"1+1"}}',
                'THOUGHT: done\nCALL: {"tool":"answer","args":{"text":"2"}}',
            }),
            tools = reg, tokenizer = { encode = function(_, s) return { #s } end },
        })
        local answer, _, steps = a:run("one plus one")
        eq("recovers from a parse error", answer, "2")
        eq("parse error is a named status", steps[1].status, "parse_error")
    end

    do  -- tool error becomes an observation
        local a = Agent.new({
            backend = scripted({
                'THOUGHT: bad\nCALL: {"tool":"calc","args":{"expr":"not math"}}',
                'THOUGHT: done\nCALL: {"tool":"answer","args":{"text":"recovered"}}',
            }),
            tools = reg, tokenizer = { encode = function(_, s) return { #s } end },
        })
        local answer, _, steps = a:run("x")
        eq("tool error is a named status", steps[1].status, "tool_error")
        check("error text reaches the model", (steps[1].observation or ""):match("[Ee]rror") ~= nil,
            steps[1].observation)
        eq("recovers from a tool error", answer, "recovered")
    end

    do  -- loop detection
        local same = 'THOUGHT: again\nCALL: {"tool":"calc","args":{"expr":"1+1"}}'
        local a = Agent.new({
            backend = scripted({ same, same, same, same, same, same, same, same }),
            tools = reg, tokenizer = { encode = function(_, s) return { #s } end },
            repeat_threshold = 3, max_steps = 8,
        })
        local _, _, steps = a:run("spin")
        local saw_repeat = false
        for _, s in ipairs(steps) do
            if s.status == "repeat" then saw_repeat = true end
        end
        check("loop detector fires", saw_repeat)
    end

    do  -- step ceiling
        local a = Agent.new({
            backend = scripted({}),
            tools = reg, tokenizer = { encode = function(_, s) return { #s } end },
            max_steps = 3,
        })
        local _, reason = a:run("x")
        check("terminates on its own", reason == "answer" or reason == "max_steps", reason)
    end
end

--------------------------------------------------------------------------
group("http")
--------------------------------------------------------------------------
-- No network, no key, no tokens spent. What is covered here is the request
-- shape and every branch of response handling; what is NOT covered is
-- whether the live API accepts the request. The README says so plainly.
--
-- These are the branches a happy-path live call would never reach anyway:
-- thinking blocks arriving before the text, a refusal at HTTP 200, an error
-- object, a transport failure.
do
    local ok_http, http = pcall(require, "backend.http")
    if ok_http and http.new then
        local sent
        local function transport_returning(raw, ok)
            return function(body_json)
                sent = body_json
                return ok ~= false, raw, 0
            end
        end

        -- Request shape
        local be = http.new({ transport = transport_returning(
            '{"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn"}') })
        be:complete("TASK: add 2 and 2")
        local req = trace.decode(sent or "")
        check("request decodes", req ~= nil)
        if req then
            eq("model defaults to opus 5", req.model, "claude-opus-5")
            eq("temperature is deterministic", req.temperature, 0)
            check("max_tokens is set", type(req.max_tokens) == "number")
            check("system prompt carries the format", (req.system or ""):match("CALL:") ~= nil)
            check("stop sequences sent", type(req.stop_sequences) == "table")
            check("one user message", req.messages and #req.messages == 1
                and req.messages[1].role == "user")
            check("prompt reaches the message",
                (req.messages[1].content or ""):match("add 2 and 2") ~= nil)
            -- Assistant prefill is rejected on current models; the backend
            -- must never send one.
            local has_assistant = false
            for _, m in ipairs(req.messages or {}) do
                if m.role == "assistant" then has_assistant = true end
            end
            check("no assistant prefill", not has_assistant)
        end

        -- Thinking blocks precede the text on Opus-tier models; taking
        -- content[1] blindly would return an empty string.
        local text = http.parse_response(
            '{"content":[{"type":"thinking","thinking":"hmm"},' ..
            '{"type":"text","text":"THOUGHT: ok"},{"type":"text","text":"\\nCALL: {}"}],' ..
            '"stop_reason":"end_turn","usage":{"output_tokens":7,"input_tokens":50},' ..
            '"model":"claude-opus-5"}')
        eq("skips thinking blocks and joins text",
           text, "THOUGHT: ok\nCALL: {}")

        local _, meta = http.parse_response(
            '{"content":[{"type":"text","text":"x"}],"stop_reason":"end_turn",' ..
            '"usage":{"output_tokens":7,"input_tokens":50},"model":"claude-opus-5"}')
        eq("reports output tokens", meta.tokens, 7)
        eq("reports input tokens", meta.input_tokens, 50)
        eq("reports the serving model", meta.model, "claude-opus-5")
        eq("http is never constrained", meta.constrained, false)

        -- A refusal is HTTP 200 with stop_reason "refusal", not an error.
        local r_ok, r_err = pcall(http.parse_response,
            '{"content":[],"stop_reason":"refusal",' ..
            '"stop_details":{"type":"refusal","category":"cyber"}}')
        check("refusal raises", not r_ok)
        check("refusal names the category",
            tostring(r_err):match("cyber") ~= nil, r_err)

        -- An API error object.
        local e_ok, e_err = pcall(http.parse_response,
            '{"type":"error","error":{"type":"authentication_error","message":"bad key"}}')
        check("api error raises", not e_ok)
        check("api error surfaces the message",
            tostring(e_err):match("bad key") ~= nil, e_err)

        -- Malformed body (an HTML error page, say).
        local m_ok = pcall(http.parse_response, "<html>502</html>")
        check("malformed body raises rather than returning junk", not m_ok)

        -- Transport failure must surface, not be swallowed.
        local be2 = http.new({ transport = transport_returning("no route to host", false) })
        local t_ok, t_err = pcall(function() return be2:complete("x") end)
        check("transport failure raises", not t_ok)
        check("transport failure keeps the detail",
            tostring(t_err):match("no route to host") ~= nil, t_err)

        -- The agent loop must treat a backend failure as fatal, not crash.
        if ok_tools and ok_agent then
            local reg = tools.registry()
            reg:add(tools.calc)
            local a = Agent.new({
                backend = http.new({ transport = transport_returning("boom", false) }),
                tools = reg,
                tokenizer = { encode = function(_, s) return { #s } end },
                max_steps = 3,
            })
            local answer, reason = a:run("x")
            eq("backend failure ends the run as fatal", reason, "fatal")
            check("no answer is invented", answer == nil)
        end
    end
end

--------------------------------------------------------------------------
group("tmpfile")
--------------------------------------------------------------------------
-- The transport seam that makes the http and ollama backends testable also
-- means their curl path is never executed by those tests. It had a bug the
-- whole time: os.tmpname() on Windows returns a bare name rooted at the
-- current drive ("\s5tg."), io.open on it fails with permission denied, and
-- the backends indexed the nil handle. Every real request died before it
-- was sent.
--
-- So this tests the staging directly -- it is the part the seam hides.
do
    local ok_tmp, tmpfile = pcall(require, "tmpfile")
    if ok_tmp then
        local p1 = tmpfile.name(".req")
        local p2 = tmpfile.name(".req")
        check("names are unique", p1 ~= p2, p1 .. " == " .. p2)
        check("name carries the suffix", p1:sub(-4) == ".req", p1)

        local path, err = tmpfile.write("hello world", ".txt")
        check("writes a file that can actually be opened", path ~= nil, err)
        if path then
            eq("round-trips the contents", tmpfile.slurp(path), "hello world")
            tmpfile.remove(path)
            check("remove deletes it", tmpfile.slurp(path) == nil)
        end

        -- Bodies contain model output: quotes, newlines, braces.
        local tricky = '{"tool":"calc","args":{"expr":"2+2"}}\nsecond "line"\n'
        local p3 = tmpfile.write(tricky, ".json")
        if p3 then
            eq("survives quotes and newlines", tmpfile.slurp(p3), tricky)
            tmpfile.remove(p3)
        end

        check("slurp of a missing file returns nil",
            tmpfile.slurp(tmpfile.name(".nope")) == nil)
        -- remove() is called on cleanup paths where some handles are nil.
        local rok = pcall(tmpfile.remove, nil, nil)
        check("remove tolerates nils", rok)
    end
end

--------------------------------------------------------------------------
group("ollama")
--------------------------------------------------------------------------
-- Same fixture approach as the http group: no daemon, no model pull.
do
    local ok_ol, ollama = pcall(require, "backend.ollama")
    if ok_ol and ollama.new then
        local sent
        local function transport(raw, ok)
            return function(body) sent = body return ok ~= false, raw, 0 end
        end

        local oreg = tools.registry()
        oreg:add(tools.calc)

        local be = ollama.new({ format = false, transport = transport(
            '{"model":"qwen2.5:0.5b-instruct","response":"THOUGHT: ok\\nCALL: {}",' ..
            '"done":true,"eval_count":12,"prompt_eval_count":80}') })
        local text, meta = be:complete("TASK: x")

        local req = trace.decode(sent or "")
        check("request decodes", req ~= nil)
        if req then
            eq("default model", req.model, "qwen2.5:0.5b-instruct")
            eq("streaming off", req.stream, false)
            -- /api/chat: role-separated messages, not system+prompt fields.
            check("system turn is first",
                req.messages and req.messages[1] and req.messages[1].role == "system")
            check("free mode asks for the text format in the system turn",
                req.messages and (req.messages[1].content or ""):match("CALL:") ~= nil)
            check("the prompt is the user turn",
                req.messages and req.messages[2]
                and (req.messages[2].content or ""):match("TASK: x") ~= nil)
            check("stop sequences set", req.options and type(req.options.stop) == "table")
            eq("temperature deterministic", req.options and req.options.temperature, 0)
            check("no format schema in free mode", req.format == nil)
        end

        eq("reads the response field", text, "THOUGHT: ok\nCALL: {}")
        eq("reports eval_count as tokens", meta.tokens, 12)
        eq("reports prompt_eval_count", meta.input_tokens, 80)
        -- This backend must never claim grammar.lua was involved; the whole
        -- mechanism/capability distinction rests on it.
        eq("never reports itself as constrained", meta.constrained, false)

        -- Schema mode: the per-tool JSON schema goes with the request, and
        -- the reply is rewrapped into the repo's one text protocol.
        local be2 = ollama.new({ tools = oreg, transport = transport(
            '{"message":{"content":"{\\"thought\\":\\"mul\\",\\"call\\":' ..
            '{\\"tool\\":\\"calc\\",\\"args\\":{\\"expr\\":\\"2+2\\"}}}"},"done":true}') })
        local t2, m2 = be2:complete("x")
        local req2 = trace.decode(sent or "")
        check("schema sent by default",
            req2 and type(req2.format) == "table" and req2.format.type == "object")
        eq("schema mode is reported", m2.schema, true)
        eq("schema reply is rewrapped into the text protocol", t2,
           'THOUGHT: mul\nCALL: {"args":{"expr":"2+2"},"tool":"calc"}')
        local turn2 = protocol.parse(t2)
        check("and the rewrapped turn parses", turn2 ~= nil and turn2.tool == "calc")

        -- KEY ORDER IS LOAD-BEARING, and nothing else in the repo cares.
        -- llama.cpp compiles the schema to GBNF in declaration order, so
        -- trace.encode's alphabetical sort put `call` before `thought` and
        -- `args` before `tool` -- forcing the model to pick a tool before
        -- reasoning, and to write arguments before naming the tool. qwen2.5
        -- then re-called a tool it had already run, five times running, and
        -- read as a model too small for the job. It was the encoder.
        local raw2 = sent or ""
        local ti = raw2:find('"thought"', 1, true)
        local ci = raw2:find('"call"', 1, true)
        check("schema declares thought before call", ti and ci and ti < ci,
            "thought at " .. tostring(ti) .. ", call at " .. tostring(ci))
        local tool_i = raw2:find('"tool"', 1, true)
        local args_i = raw2:find('"args"', 1, true)
        check("schema declares tool before args", tool_i and args_i and tool_i < args_i,
            "tool at " .. tostring(tool_i) .. ", args at " .. tostring(args_i))

        -- The overwhelmingly most common first failure: model not pulled.
        local p_ok, p_err = pcall(ollama.parse_response,
            '{"error":"model \'qwen2.5:0.5b-instruct\' not found, try pulling it first"}',
            "qwen2.5:0.5b-instruct")
        check("unpulled model raises", not p_ok)
        check("and says how to fix it",
            tostring(p_err):match("ollama pull") ~= nil, p_err)

        -- Any other Ollama error.
        local g_ok, g_err = pcall(ollama.parse_response, '{"error":"out of memory"}', "m")
        check("generic error raises", not g_ok)
        check("generic error keeps the message",
            tostring(g_err):match("out of memory") ~= nil, g_err)

        -- Daemon down: curl writes nothing.
        local d_ok = pcall(ollama.parse_response, "", "m")
        check("empty body raises rather than returning junk", not d_ok)

        -- A well-formed response with no response field.
        local n_ok = pcall(ollama.parse_response, '{"done":true}', "m")
        check("missing response field raises", not n_ok)
    end
end

--------------------------------------------------------------------------
group("replay")
--------------------------------------------------------------------------
-- Every transcript in examples/ is backed by a trace in traces/. Replaying
-- one must reproduce the recorded answer exactly. This is what makes the
-- examples evidence rather than illustration: if a transcript stops
-- reproducing, this fails.
--
-- Skipped silently when the traces are absent (regenerate: lua54 record.lua).
do
    local ok_replay, replay = pcall(require, "backend.replay")
    local ok_tasks, tasklist = pcall(require, "eval.tasks.tasks")
    local ok_proto, proto = pcall(require, "protocol")

    if ok_replay and ok_tasks and ok_proto and ok_tools and ok_agent then
        local real_render = proto.render_system
        proto.render_system = function(t, task) return real_render(t, task, true) end

        local approx = { encode = function(_, s)
            local o = {}
            for k = 1, math.max(1, math.ceil(#s / 4)) do o[k] = k end
            return o
        end }

        local found = 0
        for _, task in ipairs(tasklist.tasks) do
            local path = "traces/" .. task.id .. ".jsonl"
            local fh = io.open(path, "r")
            if fh then
                fh:close()
                found = found + 1
                local reg = tools.registry()
                for _, n in ipairs(task.tools) do reg:add(tools[n]) end
                local a = Agent.new({
                    backend = replay.new({ path = path }),
                    tools = reg,
                    tokenizer = approx,
                    max_steps = 10,
                    max_tokens = task.budget or 4096,
                    approve = function() return task.approve == true end,
                })
                local ran, answer, _, steps = pcall(function()
                    local x, y, z = a:run(task.prompt)
                    return x, y, z
                end)
                check("replays " .. task.id, ran, answer)
                if ran then
                    local passed, why = task.check(answer, steps or {}, { evicted = true })
                    check("replay of " .. task.id .. " still passes its check",
                        passed, why)
                end
            end
        end
        proto.render_system = real_render
        if found == 0 then
            io.write("  (no traces on disk; run: lua54 record.lua)\n")
        end
    end
end

--------------------------------------------------------------------------
io.write(string.format("\n%d passed, %d failed\n", pass, fail))
os.exit(fail == 0 and 0 or 1)
