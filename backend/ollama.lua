-- backend/ollama.lua - A local instruction-following model, via Ollama.
--
-- The gap this fills: backend/local.lua runs a model small enough to read
-- end to end, and that model cannot choose a tool. backend/http.lua runs a
-- model that can, and needs an account. Ollama is the third corner -- local,
-- free, no key -- and the first configuration in which this repo's eval
-- tasks can actually be completed.
--
--   ollama pull qwen2.5:0.5b-instruct
--   lua54 main.lua "What is 4871 * 209?" --backend ollama
--
--
-- WHAT THIS BACKEND DOES *NOT* DO, AND WHY IT MATTERS
--
-- Ollama owns its own decoder, so grammar.lua is not in the loop here. The
-- constraint that makes a 15M model emit valid JSON is simply absent; this
-- backend relies on the model being competent enough to follow the format,
-- which is exactly the capability the grammar substitutes for.
--
-- So the two are measuring different things, and the README says so:
--
--   backend/local.lua + grammar.lua  ->  the MECHANISM   (how tool calls work)
--   backend/ollama.lua               ->  the CAPABILITY  (can the task be done)
--
-- Reporting an Ollama task success as evidence for constrained decoding
-- would be a category error. It is evidence about the loop.
--
-- Ollama can apply a JSON schema itself (`format`), which is llama.cpp's
-- GBNF underneath -- the same idea as grammar.lua, one abstraction layer
-- down. It is ON by default here; --ollama-free turns it off, and the
-- contrast is the interesting comparison: our grammar on a 15M against
-- theirs on a 0.5B.

local trace    = require('trace')
local tmpfile  = require('tmpfile')
local protocol = require('protocol')

local backend = {}

local Ollama = {}
Ollama.__index = Ollama

local DEFAULT_HOST = "http://localhost:11434"

-- opts:
--   model       default qwen2.5:0.5b-instruct
--   host        default http://localhost:11434
--   temperature default 0
--   format      default true; the per-tool JSON schema (GBNF) is applied
--   tools       tool registry, required when format is on
--   transport   test seam, same contract as backend/http.lua
function backend.new(opts)
    opts = opts or {}
    -- Schema-constrained by default. A 0.5B model does not reliably follow
    -- a format instruction, and the measurement in turn_schema's comment is
    -- why: free decoding produced an unusable call every time. Pass
    -- format=false to reproduce that arm.
    local format = opts.format
    if format == nil then format = true end
    if format and not opts.tools then
        error("backend/ollama: schema mode needs the tool registry", 0)
    end
    return setmetatable({
        model = opts.model or "qwen2.5:0.5b-instruct",
        host = opts.host or os.getenv("OLLAMA_HOST") or DEFAULT_HOST,
        temperature = opts.temperature or 0,
        format = format,
        tools = opts.tools,
        transport = opts.transport,
    }, Ollama)
end

-- Swap the tool set; the schema is rebuilt per request so this is just a
-- field. Mirrors backend/local.lua's set_tools so eval/run.lua can treat
-- the two the same way.
function Ollama:set_tools(reg)
    self.tools = reg
end

-- The JSON schema for one turn, built from the tool registry.
--
-- This is grammar.lua's structure expressed in JSON Schema: one branch per
-- tool, each pinning `tool` to a constant and requiring exactly that tool's
-- arguments. llama.cpp compiles it to GBNF and masks logits with it, which
-- is the same mechanism one abstraction layer down.
--
-- It is not decoration. Measured on qwen2.5:0.5b-instruct, "What is
-- 4871 * 209?":
--
--   unconstrained  {"text":"4871 * 209 = 1049999"}
--                  no tool, invented argument name, invented product
--   constrained    {"thought":"...","call":{"tool":"calc",
--                                           "args":{"expr":"4871 * 209"}}}
--
-- Same model, same prompt. The schema is what makes the call usable -- the
-- same claim grammar.lua makes about stories15M, holding at 33x the size.
local function turn_schema(tools)
    local branches = {}
    for _, tool in ipairs(tools:list()) do
        local props, required = {}, {}
        for _, a in ipairs(tool.args or {}) do
            if a.required then
                props[a.name] = { type = "string" }
                required[#required + 1] = a.name
            end
        end
        branches[#branches + 1] = trace.ordered({
            type = "object",
            properties = trace.ordered({
                tool = { const = tool.name },
                args = { type = "object", properties = props, required = required },
            }, { "tool", "args" }),
            required = { "tool", "args" },
        }, { "type", "properties", "required" })
    end
    -- The property ORDER is load-bearing, not cosmetic: GBNF generates in
    -- declaration order, so this is what makes the model think before it
    -- picks a tool, and name the tool before filling its arguments.
    return trace.ordered({
        type = "object",
        properties = trace.ordered({
            thought = { type = "string" },
            call = { oneOf = branches },
        }, { "thought", "call" }),
        required = { "thought", "call" },
    }, { "type", "properties", "required" })
end

-- /api/chat, not /api/generate: the chat endpoint applies the model's own
-- template, which is what an instruct model expects. (Tested both -- the
-- endpoint was not what fixed format adherence, the schema was, but the
-- chat endpoint is still the correct one to use here.)
function Ollama:build_request(prompt, opts)
    local system
    if self.format then
        -- The schema carries the shape, so the system prompt only has to
        -- carry the job. Repeating format instructions a grammar already
        -- enforces wastes context and gives a small model more to ignore.
        --
        -- What it does have to push on is the temptation to answer from
        -- memory. A first draft ending "...use the answer tool when you are
        -- done" made qwen2.5:0.5b answer "4871 * 209" immediately instead of
        -- calling calc: the schema guarantees a well-formed call and has
        -- nothing to say about whether it was the right one. The same
        -- syntax/semantics split grammar.lua demonstrates, showing up here
        -- as a prompt problem rather than a decoding one.
        system = table.concat({
            "You are an agent working through a task one tool call at a time.",
            "Do not answer from your own knowledge if a tool can get the "
            .. "answer -- your arithmetic and your memory of file contents "
            .. "are both unreliable.",
            "Use the answer tool only after a tool has given you the result.",
        }, "\n")
    else
        system = table.concat({
            "You are an agent driving a tool loop. Reply with exactly two "
            .. "lines and nothing else:",
            protocol.THOUGHT_PREFIX .. "<one sentence>",
            protocol.CALL_PREFIX .. '{"tool":"<name>","args":{"<key>":"<value>"}}',
            "The CALL line must be a single line of valid JSON. Every "
            .. "argument value is a string. Do not use a code fence.",
        }, "\n")
    end

    -- Render the transcript as alternating turns when the agent supplies
    -- them. An instruct model is trained on assistant/user alternation, and
    -- the difference is not cosmetic: with the whole history flattened into
    -- one user message, qwen2.5:0.5b read back its own RESULT: 1018039 and
    -- called calc with the same expression again, five times running.
    local messages = { { role = "system", content = system } }
    if opts and opts.steps and #opts.steps > 0 then
        messages[#messages + 1] = { role = "user", content = opts.task_block }
        for _, s in ipairs(opts.steps) do
            if s.call then
                -- Same order as the schema. The history is the model's own
                -- prior turns; if it is shaped differently from what the
                -- grammar will accept next, the model imitates the history.
                messages[#messages + 1] = { role = "assistant",
                    content = trace.encode(trace.ordered({
                        thought = s.thought or "",
                        call = trace.ordered({
                            tool = s.call.tool, args = s.call.args or {},
                        }, { "tool", "args" }),
                    }, { "thought", "call" })) }
            end
            if s.observation then
                messages[#messages + 1] = { role = "user",
                    content = protocol.OBS_PREFIX .. s.observation }
            end
        end
    else
        messages[#messages + 1] = { role = "user", content = prompt }
    end

    local body = {
        model = self.model,
        stream = false,
        messages = messages,
        options = { temperature = self.temperature },
    }
    -- Stop strings only make sense for the free-text protocol; under a
    -- schema the response is a single JSON object.
    if not self.format then
        body.options.stop = protocol.STOP
    else
        body.format = turn_schema(self.tools)
    end
    return body
end

-- /api/chat returns {"message":{"content":"..."},...}; /api/generate used
-- {"response":"..."}. Accept either so the backend does not break if the
-- endpoint is switched back. Errors arrive as {"error":"..."}, most often
-- for a model that was never pulled -- the single most likely first failure,
-- so it gets an actionable message.
function backend.parse_response(raw, model, schema_mode)
    local resp, derr = trace.decode(raw)
    if not resp then
        error("could not parse the Ollama response: " .. tostring(derr), 0)
    end
    if resp.error then
        local msg = tostring(resp.error)
        if msg:lower():find("not found") or msg:lower():find("try pulling") then
            error(string.format(
                "Ollama does not have %q. Run:  ollama pull %s",
                tostring(model), tostring(model)), 0)
        end
        error("Ollama: " .. msg, 0)
    end

    local content = resp.message and resp.message.content or resp.response
    if type(content) ~= "string" then
        error("Ollama returned no message content", 0)
    end

    local meta = {
        tokens = resp.eval_count,
        input_tokens = resp.prompt_eval_count,
        model = resp.model,
        constrained = false,   -- never grammar.lua; see the header
        schema = schema_mode or false,
    }

    -- Under a schema the reply is one JSON object, not the two-line text
    -- protocol. Rewrap it so protocol.parse -- and therefore the loop, the
    -- trace format and every transcript -- stays identical across backends.
    -- The adaptation belongs here, in the thing that knows its own wire
    -- format, rather than as a special case in the parser.
    if schema_mode then
        local turn = trace.decode(content)
        if type(turn) == "table" and type(turn.call) == "table" then
            local thought = type(turn.thought) == "string" and turn.thought or ""
            return protocol.THOUGHT_PREFIX .. thought .. "\n"
                .. protocol.CALL_PREFIX .. trace.encode({
                    tool = turn.call.tool, args = turn.call.args or {},
                }), meta
        end
        -- Schema was requested but the reply did not match it. Hand the raw
        -- text back and let protocol.parse produce the error the model sees;
        -- inventing a call here would hide a real failure.
    end

    return content, meta
end

function Ollama:curl(body_json)
    -- No API key, so no header file needed -- but the body still goes via a
    -- file rather than the command line, because it contains model output
    -- with quotes and newlines.
    local req_path, err = tmpfile.write(body_json, ".req")
    if not req_path then return false, err, -1 end
    local out_path = tmpfile.name(".out")

    local cmd = string.format(
        'curl -sS --fail-with-body -X POST %s/api/chat '
        .. '-H "content-type: application/json" -d @"%s" -o "%s"',
        self.host, req_path, out_path)
    local ok, _, code = os.execute(cmd)

    local raw = tmpfile.slurp(out_path) or ""
    tmpfile.remove(req_path, out_path)

    if not ok and raw == "" then
        return false, string.format(
            "cannot reach Ollama at %s -- is it running? (`ollama serve`)",
            self.host), code
    end
    return ok, raw, code
end

function Ollama:complete(prompt, opts)
    local body_json = trace.encode(self:build_request(prompt, opts))

    local ok, raw, code
    if self.transport then
        ok, raw, code = self.transport(body_json)
    else
        ok, raw, code = self:curl(body_json)
    end
    if not ok then
        error(string.format("Ollama request failed (exit %s): %s",
            tostring(code), tostring(raw):sub(1, 400)), 0)
    end
    return backend.parse_response(raw, self.model, self.format)
end

return backend
