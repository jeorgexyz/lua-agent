-- trace.lua - JSONL record and replay, plus the minimal JSON it needs.
--
-- The highest-value file in the repo for its size. Every step is appended as
-- one JSON line; backend/replay.lua reads those lines back. That single fact
-- buys:
--   - a test suite that needs no API key and no network
--   - examples/ transcripts that are reproducible rather than illustrative
--   - a debugging story: re-run a failure exactly, as many times as you like
--
-- No dependencies is load-bearing here, same as repo 1, so the JSON codec is
-- written out rather than pulled in. It only has to handle the step schema.

local trace = {}

--------------------------------------------------------------------------
-- JSON encode
--------------------------------------------------------------------------

local ESCAPES = {
    ['"'] = '\\"', ['\\'] = '\\\\', ['\b'] = '\\b', ['\f'] = '\\f',
    ['\n'] = '\\n', ['\r'] = '\\r', ['\t'] = '\\t',
}

local function escape_str(s)
    return (s:gsub('[%c"\\]', function(c)
        return ESCAPES[c] or string.format('\\u%04x', c:byte())
    end))
end

-- Arrays are tables whose keys are exactly 1..n. Everything else, including
-- the empty table, encodes as an object -- empty `args` must be `{}`.
local function is_array(t)
    local n = 0
    for _ in pairs(t) do n = n + 1 end
    for i = 1, n do
        if t[i] == nil then return false end
    end
    return n > 0
end

-- Pin an object's key order for encoding.
--
-- Key order is semantically meaningless in JSON, which is why encode sorts
-- alphabetically by default: byte-stable traces that diff cleanly.
--
-- It stops being meaningless the moment the JSON is a SCHEMA that gets
-- compiled to a grammar. llama.cpp emits object properties in declaration
-- order, so the sort silently rewrote backend/ollama.lua's turn schema from
-- {thought, call} to {call, thought} and from {tool, args} to {args, tool} --
-- forcing the model to choose a tool before writing its reasoning, and to
-- write arguments before naming the tool they belonged to. qwen2.5 then
-- re-called a tool it had already run, five times in a row, and looked like
-- a model too small for the job.
--
--   trace.ordered({ thought = "...", call = {...} }, { "thought", "call" })
--
-- Keys not listed are appended in sorted order, so this never drops data.
function trace.ordered(tbl, order)
    return setmetatable(tbl, { __jsonorder = order })
end

function trace.encode(v)
    local ty = type(v)
    if v == nil then
        return "null"
    elseif ty == "boolean" then
        return tostring(v)
    elseif ty == "number" then
        -- %.14g round-trips a double without trailing float noise
        return (v % 1 == 0 and math.abs(v) < 1e15)
            and string.format("%d", v)
            or string.format("%.14g", v)
    elseif ty == "string" then
        return '"' .. escape_str(v) .. '"'
    elseif ty == "table" then
        local out = {}
        if is_array(v) then
            for i = 1, #v do out[#out + 1] = trace.encode(v[i]) end
            return "[" .. table.concat(out, ",") .. "]"
        end
        -- Sort keys so a re-encoded trace is byte-stable and diffable --
        -- unless the caller pinned an order with trace.ordered.
        local keys = {}
        local pinned = getmetatable(v)
        pinned = pinned and pinned.__jsonorder
        if pinned then
            for _, k in ipairs(pinned) do
                if v[k] ~= nil then keys[#keys + 1] = k end
            end
            -- Anything not named in the order still has to be emitted.
            local named = {}
            for _, k in ipairs(pinned) do named[k] = true end
            local rest = {}
            for k in pairs(v) do
                if not named[k] then rest[#rest + 1] = k end
            end
            table.sort(rest, function(a, b) return tostring(a) < tostring(b) end)
            for _, k in ipairs(rest) do keys[#keys + 1] = k end
        else
            for k in pairs(v) do keys[#keys + 1] = k end
            table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
        end
        for _, k in ipairs(keys) do
            out[#out + 1] = '"' .. escape_str(tostring(k)) .. '":' .. trace.encode(v[k])
        end
        return "{" .. table.concat(out, ",") .. "}"
    end
    error("cannot encode " .. ty)
end

--------------------------------------------------------------------------
-- JSON decode -- recursive descent, enough for the schema above
--------------------------------------------------------------------------

local Parser = {}
Parser.__index = Parser

function Parser:err(msg)
    error(string.format("json: %s at byte %d", msg, self.pos), 0)
end

function Parser:skip_ws()
    local _, e = self.s:find("^[ \t\r\n]*", self.pos)
    self.pos = e + 1
end

function Parser:peek()
    return self.s:sub(self.pos, self.pos)
end

function Parser:expect(c)
    if self:peek() ~= c then self:err("expected " .. c) end
    self.pos = self.pos + 1
end

local UNESCAPES = {
    ['"'] = '"', ['\\'] = '\\', ['/'] = '/', b = '\b',
    f = '\f', n = '\n', r = '\r', t = '\t',
}

function Parser:parse_string()
    self:expect('"')
    local buf = {}
    while true do
        local c = self:peek()
        if c == "" then self:err("unterminated string") end
        self.pos = self.pos + 1
        if c == '"' then
            return table.concat(buf)
        elseif c == '\\' then
            local esc = self:peek()
            self.pos = self.pos + 1
            if esc == "u" then
                local hex = self.s:sub(self.pos, self.pos + 3)
                if not hex:match("^%x%x%x%x$") then self:err("bad unicode escape") end
                self.pos = self.pos + 4
                local cp = tonumber(hex, 16)
                buf[#buf + 1] = (cp < 128) and string.char(cp) or utf8.char(cp)
            elseif UNESCAPES[esc] then
                buf[#buf + 1] = UNESCAPES[esc]
            else
                self:err("bad escape " .. esc)
            end
        else
            buf[#buf + 1] = c
        end
    end
end

function Parser:parse_value()
    self:skip_ws()
    local c = self:peek()
    if c == '"' then
        return self:parse_string()
    elseif c == "{" then
        self.pos = self.pos + 1
        local obj = {}
        self:skip_ws()
        if self:peek() == "}" then self.pos = self.pos + 1 return obj end
        while true do
            self:skip_ws()
            local k = self:parse_string()
            self:skip_ws()
            self:expect(":")
            obj[k] = self:parse_value()
            self:skip_ws()
            local d = self:peek()
            self.pos = self.pos + 1
            if d == "}" then return obj end
            if d ~= "," then self:err("expected comma or close-brace") end
        end
    elseif c == "[" then
        self.pos = self.pos + 1
        local arr = {}
        self:skip_ws()
        if self:peek() == "]" then self.pos = self.pos + 1 return arr end
        while true do
            arr[#arr + 1] = self:parse_value()
            self:skip_ws()
            local d = self:peek()
            self.pos = self.pos + 1
            if d == "]" then return arr end
            if d ~= "," then self:err("expected comma or close-bracket") end
        end
    elseif self.s:find("^true", self.pos) then
        self.pos = self.pos + 4 return true
    elseif self.s:find("^false", self.pos) then
        self.pos = self.pos + 5 return false
    elseif self.s:find("^null", self.pos) then
        self.pos = self.pos + 4 return nil
    else
        local num = self.s:match("^%-?%d+%.?%d*[eE]?[-+]?%d*", self.pos)
        if not num or num == "" then self:err("unexpected character " .. c) end
        self.pos = self.pos + #num
        return tonumber(num)
    end
end

-- Returns value, err. Never raises: callers feed the error text straight back
-- to the model as an observation, which is how parse recovery works.
function trace.decode(str)
    local p = setmetatable({ s = str, pos = 1 }, Parser)
    local ok, result = pcall(function()
        local v = p:parse_value()
        p:skip_ws()
        if p.pos <= #p.s then p:err("trailing content") end
        return v
    end)
    if not ok then return nil, result end
    return result
end

--------------------------------------------------------------------------
-- JSONL trace files
--------------------------------------------------------------------------

local Trace = {}
Trace.__index = Trace

-- mode: "record" | "off"
function trace.open(path, mode)
    local self = setmetatable({ path = path, mode = mode or "off", n = 0 }, Trace)
    if self.mode == "record" then
        if not path then error("trace.open: record mode needs a path") end
        local fh, err = io.open(path, "w")
        if not fh then error("trace.open: " .. tostring(err)) end
        self.fh = fh
    end
    return self
end

-- One line per step. Schema (stable; examples/ depend on it):
--   { step=, prompt=, raw=, thought=, call={tool=,args=}, observation=,
--     status=, tokens={prompt=,completion=,window=}, ms= }
function Trace:write(record)
    self.n = self.n + 1
    if not self.fh then return end
    self.fh:write(trace.encode(record), "\n")
    self.fh:flush()  -- a crashed run must still leave a readable trace
end

function Trace:close()
    if self.fh then self.fh:close() self.fh = nil end
end

-- Read a trace back into a step list.
function trace.read(path)
    local fh, err = io.open(path, "r")
    if not fh then error("trace.read: " .. tostring(err)) end
    local steps, lineno = {}, 0
    for line in fh:lines() do
        lineno = lineno + 1
        if line:match("%S") then
            local rec, derr = trace.decode(line)
            if not rec then
                fh:close()
                error(string.format("trace.read: %s:%d: %s", path, lineno, derr))
            end
            steps[#steps + 1] = rec
        end
    end
    fh:close()
    return steps
end

return trace
