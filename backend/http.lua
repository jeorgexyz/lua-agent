-- backend/http.lua - Anthropic Messages API.
--
-- The capability demo: same loop, same protocol, same trace format, but a
-- model that can actually finish the task. Swapping this for
-- backend/local.lua is one flag in main.lua, which is the argument that the
-- loop -- not the model -- is the artifact.
--
--
-- WHAT THIS BACKEND DELIBERATELY DOES NOT DO
--
-- It does not use the provider's native tool-use API. It sends the same text
-- protocol from protocol.lua that the local backend uses, and parses the
-- reply with the same parser.
--
-- In production you should do the opposite: the native API is more reliable
-- and you do not have to maintain a grammar. It is avoided here because
-- native tool use hides the exact mechanism this repo exists to show, and
-- because sharing one protocol is what makes the backends interchangeable.
-- Read grammar.lua to see what the native API is doing on your behalf.
--
-- Two consequences of talking to a current model that are worth knowing:
--
--   * Assistant prefill is rejected on current models, so this backend
--     cannot scaffold "THOUGHT: " the way backend/local.lua does. The format
--     has to be requested in the system prompt and honoured voluntarily --
--     which a capable model does, and which is exactly the capability the
--     grammar substitutes for on a 15M model.
--   * Thinking is on by default on Opus-tier models, so `content` is an
--     array that may hold thinking blocks before the text. Filter by type;
--     never take content[1] blindly.
--
--
-- Requires ANTHROPIC_API_KEY and curl on PATH. No Lua HTTP library, which
-- keeps the zero-dependency property intact.

local trace    = require('trace')
local tmpfile  = require('tmpfile')
local protocol = require('protocol')

local backend = {}

local Http = {}
Http.__index = Http

local ENDPOINT = "https://api.anthropic.com/v1/messages"
local API_VERSION = "2023-06-01"

-- opts:
--   model       default claude-opus-5
--   max_tokens  default 1024 (a turn here is one thought and one call)
--   temperature default 0
--   api_key     defaults to $ANTHROPIC_API_KEY
-- opts.transport lets a test supply the HTTP call.
--   transport(body_json, headers) -> ok, raw_response_string, exit_code
-- Defaults to curl. This exists because the two things most likely to be
-- wrong here -- the shape of the request and the parsing of the response --
-- are pure functions of data, and testing them should not require a network,
-- an account, or spending anyone's tokens.
function backend.new(opts)
    opts = opts or {}
    local transport = opts.transport
    local key = opts.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not transport and (not key or key == "") then
        error("backend/http: set ANTHROPIC_API_KEY (or pass api_key)", 0)
    end
    return setmetatable({
        key = key or "test-key",
        transport = transport,
        model = opts.model or "claude-opus-5",
        max_tokens = opts.max_tokens or 1024,
        temperature = opts.temperature or 0,
    }, Http)
end

-- The system prompt carries the format contract. On the local backend the
-- grammar enforces this; here it is a request the model honours.
local SYSTEM = table.concat({
    "You are an agent driving a tool loop. Reply with exactly two lines "
    .. "and nothing else:",
    protocol.THOUGHT_PREFIX .. "<one sentence>",
    protocol.CALL_PREFIX .. '{"tool":"<name>","args":{"<key>":"<value>"}}',
    "The CALL line must be a single line of valid JSON. Every argument "
    .. "value is a string. Do not wrap it in a code fence.",
}, "\n")

-- Build the request body. Pure; separated so it can be asserted against.
function Http:build_request(prompt)
    return {
        model = self.model,
        max_tokens = self.max_tokens,
        temperature = self.temperature,
        system = SYSTEM,
        stop_sequences = protocol.STOP,
        messages = { { role = "user", content = prompt } },
    }
end

-- Turn a raw response body into text + meta, or raise with a message the
-- agent loop will surface. Pure; every branch here is a real failure mode of
-- the Messages API and none of them are reachable by a happy-path test.
function backend.parse_response(raw)
    local resp, derr = trace.decode(raw)
    if not resp then
        error("could not parse the API response: " .. tostring(derr), 0)
    end
    if resp.error then
        error(string.format("API %s: %s",
            tostring(resp.error.type), tostring(resp.error.message)), 0)
    end

    -- A policy decline arrives as HTTP 200 with stop_reason "refusal", not
    -- as an error status. Check it before reading content.
    if resp.stop_reason == "refusal" then
        local why = resp.stop_details and resp.stop_details.category or "unspecified"
        error("the model declined this request (" .. tostring(why) .. ")", 0)
    end

    -- content is an array and may hold thinking blocks before the text --
    -- thinking is on by default on Opus-tier models -- so filter by type
    -- rather than taking content[1].
    local parts = {}
    for _, block in ipairs(resp.content or {}) do
        if block.type == "text" then parts[#parts + 1] = block.text end
    end

    return table.concat(parts), {
        tokens = resp.usage and resp.usage.output_tokens,
        input_tokens = resp.usage and resp.usage.input_tokens,
        stop_reason = resp.stop_reason,
        model = resp.model,
        constrained = false,
    }
end

-- The default transport.
--
-- The request goes through files rather than the command line: the prompt
-- contains model output, newlines and quotes, and interpolating that into a
-- shell command is both fragile and a command-injection hole. The API key
-- goes in a header file for the same reason -- as an argument it would be
-- visible in the process list.
function Http:curl(body_json)
    local req_path, err = tmpfile.write(body_json, ".req")
    if not req_path then return false, err, -1 end

    local hdr_path, herr = tmpfile.write(table.concat({
        "x-api-key: " .. self.key,
        "anthropic-version: " .. API_VERSION,
        "content-type: application/json",
    }, "\n") .. "\n", ".hdr")
    if not hdr_path then
        tmpfile.remove(req_path)
        return false, herr, -1
    end

    local out_path = tmpfile.name(".out")
    local cmd = string.format(
        'curl -sS --fail-with-body -X POST %s -H @"%s" -d @"%s" -o "%s"',
        ENDPOINT, hdr_path, req_path, out_path)
    local ok, _, code = os.execute(cmd)

    local raw = tmpfile.slurp(out_path) or ""
    tmpfile.remove(req_path, hdr_path, out_path)

    return ok, raw, code
end

function Http:complete(prompt, opts)
    local body_json = trace.encode(self:build_request(prompt))

    local ok, raw, code
    if self.transport then
        ok, raw, code = self.transport(body_json)
    else
        ok, raw, code = self:curl(body_json)
    end

    if not ok then
        error(string.format("request failed (exit %s): %s",
            tostring(code), tostring(raw):sub(1, 400)), 0)
    end
    return backend.parse_response(raw)
end

return backend
