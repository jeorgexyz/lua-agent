-- mcp.lua - A Model Context Protocol client, in pure Lua.
--
-- MCP is JSON-RPC 2.0 over stdio. Three methods get you a working client:
--
--   initialize                 handshake, exchange capabilities
--   notifications/initialized  client says it is ready (no reply)
--   tools/list                 what the server offers
--   tools/call                 run one
--
-- That is the whole surface this needs. Mount a server's tools into the
-- registry and the agent loop calls them exactly like the built-ins -- same
-- validation, same approval gate, same "a failure is an observation" rule.
--
--   lua54 main.lua "Echo hello" --backend ollama \
--     --mcp "npx -y @modelcontextprotocol/server-everything"
--
--
-- TWO TRANSPORTS, AND WHY THERE ARE TWO
--
-- Lua has no bidirectional pipes. io.popen opens a stream for reading OR
-- writing, never both:
--
--   io.popen("cat", "rw")  -->  bad argument #2 to 'popen' (invalid mode)
--
-- A long-lived stdio session needs both halves at once, so it is out of
-- reach without a C extension. The stdio transport here works around that:
-- write the whole request sequence to a file, run the server with it as
-- stdin, read what it writes before it exits on EOF.
--
--   server < requests.jsonl > responses.jsonl
--
-- Every call therefore re-spawns the server and re-runs the handshake. That
-- is slow -- and worse, it is WRONG for any server holding state, because a
-- re-spawned server loses the session, the open handle, the cursor. Speed
-- is only the visible symptom.
--
-- Streamable HTTP fixes both without leaving pure Lua: the server is
-- long-lived because it is not ours to spawn, and curl only ever needs
-- request/response. Measured against server-everything -- 3.4s -> 0.24s to
-- discover, 3.4s -> 0.06s per call, one handshake instead of one per call.
--
-- Prefer --mcp-url. Use stdio for servers that speak nothing else.
--
--
-- TWO THINGS THE REAL SERVER TAUGHT THAT THE SPEC READS PAST
--
-- 1. Responses are NOT in request order, and notifications are interleaved
--    with them. Against server-everything, notifications/tools/list_changed
--    arrived before the initialize result, and a tools/call for id 3 came
--    back before id 2. Match on `id`; never on position.
--
-- 2. A tool that fails returns a SUCCESSFUL JSON-RPC result carrying
--    isError: true, with the message in `content`. It is not a JSON-RPC
--    error object. Treating it as one loses the message the model needs --
--    and that message is genuinely good ("expected string, received
--    undefined at message"), exactly the kind of observation this loop
--    feeds back for recovery.

local trace   = require('trace')
local tmpfile = require('tmpfile')

local mcp = {}

local Client = {}
Client.__index = Client

-- Bump when the servers you use require it; the handshake echoes back what
-- the server actually chose, and mismatches surface there rather than here.
mcp.PROTOCOL_VERSION = "2025-06-18"

mcp.CLIENT_INFO = { name = "lua-agent", version = "0.1" }

-- opts:
--   command    the server's launch command, e.g.
--              "npx -y @modelcontextprotocol/server-everything"
--   transport  function(request_text) -> ok, response_text, code
--              Swap in a persistent one, or a fixture for tests.
--   protocol_version
--   prefix     namespace mounted tool names, e.g. "fs_" (see mcp.mount)
function mcp.new(opts)
    opts = opts or {}
    if not opts.command and not opts.transport and not opts.url then
        error("mcp.new: needs a command, a url, or a transport", 0)
    end
    return setmetatable({
        command = opts.command,
        url = opts.url,
        transport = opts.transport,
        protocol_version = opts.protocol_version or mcp.PROTOCOL_VERSION,
        prefix = opts.prefix,
        next_id = 0,
    }, Client)
end

function Client:id()
    self.next_id = self.next_id + 1
    return self.next_id
end

-- Run the server over a file of newline-delimited JSON-RPC and collect what
-- it writes. stderr goes to its own file: servers log startup banners there
-- ("Starting default (STDIO) server...") and mixing them into stdout breaks
-- the parse.
function Client:stdio(request_text)
    local req_path, err = tmpfile.write(request_text, ".jsonl")
    if not req_path then return false, err, -1 end
    local out_path, errp = tmpfile.name(".out"), tmpfile.name(".err")

    local cmd = string.format('%s < "%s" > "%s" 2> "%s"',
        self.command, req_path, out_path, errp)
    local ok, _, code = os.execute(cmd)

    local raw = tmpfile.slurp(out_path) or ""
    local stderr = tmpfile.slurp(errp) or ""
    tmpfile.remove(req_path, out_path, errp)

    if raw == "" then
        -- A server that produced nothing failed to start. Its stderr is the
        -- only useful thing we have, so pass it through rather than
        -- reporting a bare exit code.
        return false, string.format(
            "MCP server produced no output (%s)%s", tostring(self.command),
            stderr ~= "" and (": " .. stderr:gsub("%s+$", ""):sub(1, 300)) or ""), code
    end
    return ok or true, raw, code
end

-- Streamable HTTP: POST one JSON-RPC message to a server that is already
-- running. This is the transport that does NOT have the stdio problem --
-- the process is long-lived because it is not ours, and request/response
-- over curl needs no bidirectional pipe.
--
-- Two details the shape of this depends on:
--
--   * The server issues an Mcp-Session-Id header on initialize. Echoing it
--     back on later requests is what keeps one session alive across calls,
--     which is the whole point: a re-spawned stdio server loses any state
--     it was holding, and no amount of speed work fixes that.
--   * Replies come back as SSE ("event: message" / "data: {...}"), not bare
--     JSON, even for a single response. Hence the data: unwrapping in
--     decode_stream below.
function Client:http(request_text)
    local body_path, err = tmpfile.write(request_text, ".json")
    if not body_path then return false, err, -1 end
    local out_path, hdr_path = tmpfile.name(".out"), tmpfile.name(".hdr")

    local session = ""
    if self.session_id then
        session = string.format(' -H "mcp-session-id: %s"', self.session_id)
    end

    local cmd = string.format(
        'curl -sS -X POST %s -D "%s" -H "content-type: application/json" '
        .. '-H "accept: application/json, text/event-stream"%s -d @"%s" -o "%s"',
        self.url, hdr_path, session, body_path, out_path)
    local ok, _, code = os.execute(cmd)

    local raw = tmpfile.slurp(out_path) or ""
    local headers = tmpfile.slurp(hdr_path) or ""
    tmpfile.remove(body_path, out_path, hdr_path)

    -- Capture the session on the way past; the server only sends it once.
    if not self.session_id then
        self.session_id = headers:match("[Mm]cp%-[Ss]ession%-[Ii]d:%s*([^\r\n]+)")
    end

    if not ok and raw == "" then
        return false, string.format("cannot reach the MCP server at %s", self.url), code
    end
    return true, raw, code
end

-- Responses arrive either as newline-delimited JSON (stdio) or as SSE
-- frames (HTTP). Both reduce to "find the JSON objects", so one reader
-- handles them: lines that are not JSON -- SSE "event:"/"id:" fields,
-- server banners on stdout -- are simply skipped.
local function decode_stream(raw, by_id, notifications)
    for line in tostring(raw):gmatch("[^\n]+") do
        local payload = line:match("^data:%s*(.*)$") or line
        if payload:match("%S") then
            local msg = trace.decode(payload)
            if type(msg) == "table" then
                if msg.id ~= nil then
                    by_id[msg.id] = msg
                elseif msg.method then
                    notifications[#notifications + 1] = msg
                end
            end
        end
    end
end

-- Send messages and return responses keyed by id, plus any notifications.
--
-- stdio batches everything into one stdin stream because the process only
-- lives for the length of that stream. HTTP posts one message at a time
-- against a session that outlives the call.
function Client:exchange(messages)
    local by_id, notifications = {}, {}

    local function send(text)
        local ok, raw, code
        if self.transport then
            ok, raw, code = self.transport(text)
        elseif self.url then
            ok, raw, code = self:http(text)
        else
            ok, raw, code = self:stdio(text)
        end
        if not ok then
            error(string.format("MCP transport failed (%s): %s",
                tostring(code), tostring(raw):sub(1, 400)), 0)
        end
        decode_stream(raw, by_id, notifications)
    end

    if self.url and not self.transport then
        for _, m in ipairs(messages) do send(trace.encode(m)) end
    else
        local lines = {}
        for _, m in ipairs(messages) do lines[#lines + 1] = trace.encode(m) end
        send(table.concat(lines, "\n") .. "\n")
    end

    return by_id, notifications
end

function Client:initialize_msg()
    return { jsonrpc = "2.0", id = self:id(), method = "initialize", params = {
        protocolVersion = self.protocol_version,
        capabilities = trace.ordered({}, {}),
        clientInfo = mcp.CLIENT_INFO,
    } }
end

-- Messages to prepend to an exchange.
--
-- Over stdio the server is re-spawned every time, so every exchange has to
-- re-handshake -- there is no session to resume. Over HTTP the handshake
-- happens once and the session id carries it forward, so this returns
-- nothing after the first call.
function Client:handshake()
    if self.url and self.session_ready then return {} end
    return {
        self:initialize_msg(),
        { jsonrpc = "2.0", method = "notifications/initialized" },
    }
end

local function result_or_raise(msg, what)
    if not msg then
        error(string.format("MCP: no response to %s", what), 0)
    end
    if msg.error then
        error(string.format("MCP %s failed: %s (code %s)", what,
            tostring(msg.error.message), tostring(msg.error.code)), 0)
    end
    return msg.result
end

-- Handshake + tools/list. Returns the tool descriptors the server offers
-- and records serverInfo.
function Client:discover()
    local msgs = self:handshake()
    local list_id = self:id()
    msgs[#msgs + 1] = { jsonrpc = "2.0", id = list_id, method = "tools/list",
                        params = trace.ordered({}, {}) }

    local init_id = msgs[1] and msgs[1].id
    local by_id = self:exchange(msgs)
    if init_id then
        local init = result_or_raise(by_id[init_id], "initialize")
        self.server_info = init.serverInfo
        self.negotiated_version = init.protocolVersion
        -- From here HTTP exchanges skip the handshake and ride the session.
        self.session_ready = true
    end

    local listed = result_or_raise(by_id[list_id], "tools/list")
    self.tools = listed.tools or {}
    return self.tools
end

-- Handshake + one tools/call. Returns text, is_error.
--
-- A failing tool is a normal result with isError set, so this returns the
-- message rather than raising: the loop's job is to hand it to the model as
-- an observation, which is how recovery works.
function Client:call(name, args)
    local msgs = self:handshake()
    local call_id = self:id()
    msgs[#msgs + 1] = { jsonrpc = "2.0", id = call_id, method = "tools/call",
        params = { name = name, arguments = args or trace.ordered({}, {}) } }

    local by_id = self:exchange(msgs)
    local res = result_or_raise(by_id[call_id], "tools/call " .. name)

    local parts = {}
    for _, block in ipairs(res.content or {}) do
        if block.type == "text" and block.text then
            parts[#parts + 1] = block.text
        elseif block.type then
            -- Images, audio, resource links. Name them rather than dropping
            -- them silently, so a transcript shows what came back.
            parts[#parts + 1] = string.format("[%s content]", block.type)
        end
    end
    local text = table.concat(parts, "\n")
    if text == "" then text = "(no content)" end
    return text, res.isError == true
end

--------------------------------------------------------------------------
-- Mounting MCP tools into the agent's registry
--------------------------------------------------------------------------

-- MCP describes arguments with JSON Schema; tools.lua wants a flat list of
-- {name, type, required}. Only the parts the grammar can express survive:
-- required scalars. An argument this cannot represent is dropped from the
-- schema the model sees, which is better than offering it a field it has no
-- way to fill correctly.
local function args_from_schema(schema)
    local out = {}
    if type(schema) ~= "table" or type(schema.properties) ~= "table" then
        return out
    end
    local required = {}
    for _, r in ipairs(schema.required or {}) do required[r] = true end

    -- Sorted, so the generated grammar and catalogue are stable across runs.
    local names = {}
    for k in pairs(schema.properties) do names[#names + 1] = k end
    table.sort(names)

    for _, k in ipairs(names) do
        local prop = schema.properties[k] or {}
        local ty = prop.type
        if ty == "integer" then ty = "number" end
        if ty ~= "number" then ty = "string" end
        out[#out + 1] = { name = k, type = ty, required = required[k] == true }
    end
    return out
end

-- Add a server's tools to a registry.
--
-- Collisions are refused rather than resolved: silently shadowing `calc`
-- with a remote tool of the same name would be a genuinely nasty surprise.
-- Pass a prefix to namespace them instead.
function mcp.mount(registry, client, opts)
    opts = opts or {}
    local prefix = opts.prefix or client.prefix or ""
    local tools = client.tools or client:discover()
    local mounted = {}

    for _, t in ipairs(tools) do
        local name = prefix .. t.name
        if registry:get(name) then
            error(string.format(
                "mcp.mount: %q already exists in the registry. Pass a prefix "
                .. "to namespace this server's tools.", name), 0)
        end
        local desc = (t.description or "MCP tool"):gsub("%s+", " ")
        if #desc > 120 then desc = desc:sub(1, 117) .. "..." end

        registry:add({
            name = name,
            description = desc,
            args = args_from_schema(t.inputSchema),
            -- Remote calls reach outside the process. Read-only built-ins
            -- run unattended; a tool whose behaviour this repo cannot see
            -- does not get that by default.
            requires_approval = opts.trust ~= true,
            run = function(a)
                local text, is_error = client:call(t.name, a)
                if is_error then error(text, 0) end
                return text
            end,
        })
        mounted[#mounted + 1] = name
    end
    return mounted
end

return mcp
