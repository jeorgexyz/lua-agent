-- tools.lua - Tool registry and the built-in set.
--
-- A tool is a plain table:
--   name              string, matched verbatim against the model's output
--   description       one line; this is prompt real estate, keep it short
--   args              { {name=, type="string"|"number", required=bool}, ... }
--   run               function(args) -> result_string   (may raise)
--   requires_approval bool, gates the tool behind the agent's approve callback
--   timeout_ms        number|nil

local protocol = require('protocol')

local tools = {}

--------------------------------------------------------------------------
-- Registry
--------------------------------------------------------------------------

local Registry = {}
Registry.__index = Registry

-- The answer pseudo-tool is always present: it is how the loop terminates,
-- so it must appear in the catalogue and in the grammar's enum of names.
function tools.registry()
    local self = setmetatable({ by_name = {}, order = {} }, Registry)
    self:add(protocol.answer_tool)
    return self
end

function Registry:add(tool)
    if type(tool) ~= "table" or type(tool.name) ~= "string" then
        error("tools: a tool needs a name")
    end
    if not self.by_name[tool.name] then
        self.order[#self.order + 1] = tool.name
    end
    self.by_name[tool.name] = tool
    return self
end

function Registry:get(name)
    return self.by_name[name]
end

-- Stable order, so the rendered catalogue and the compiled grammar are
-- byte-identical across runs. Traces stop diffing otherwise.
function Registry:list()
    local out = {}
    for _, n in ipairs(self.order) do out[#out + 1] = self.by_name[n] end
    return out
end

function Registry:names()
    local out = {}
    for _, n in ipairs(self.order) do out[#out + 1] = n end
    return out
end

-- Validate a parsed call before running it. Returns ok, err_or_args.
-- A validation failure is an observation, not a crash: the message goes back
-- to the model, so it is phrased for the model to act on.
function Registry:validate(name, args)
    local tool = self.by_name[name]
    if not tool then
        local avail = table.concat(self:names(), ", ")
        return false, string.format(
            "No tool named %q. Available tools: %s.", name, avail)
    end
    args = args or {}
    local clean = {}
    for _, spec in ipairs(tool.args or {}) do
        local v = args[spec.name]
        if v == nil or v == "" then
            if spec.required then
                return false, string.format(
                    "The %s tool needs a %q argument.", name, spec.name)
            end
        else
            if spec.type == "number" then
                local n = tonumber(v)
                if not n then
                    return false, string.format(
                        "The %q argument of %s must be a number, got %q.",
                        spec.name, name, tostring(v))
                end
                clean[spec.name] = n
            else
                clean[spec.name] = tostring(v)
            end
        end
    end
    return true, clean
end

-- Execute with validation, the approval gate, and error capture applied.
-- Returns ok, result_or_error, status  where status is one of
-- "continue" | "tool_error" | "timeout" | "denied".
function Registry:invoke(name, args, approve)
    local ok, clean = self:validate(name, args)
    if not ok then
        return false, "Error: " .. clean, "tool_error"
    end

    local tool = self.by_name[name]
    if tool.requires_approval then
        -- Default-deny. An agent that writes to disk because nobody wired up
        -- an approval callback is the failure mode this guards against.
        if not approve or not approve({ tool = name, args = clean }) then
            return false, string.format(
                "Denied: the user did not approve calling %s.", name), "denied"
        end
    end

    local started = os.clock()
    local ran, result = pcall(tool.run, clean)
    local elapsed_ms = (os.clock() - started) * 1000

    if not ran then
        -- Strip Lua's file:line prefix; the model cannot act on it and it
        -- costs tokens on every retry.
        local msg = tostring(result):gsub("^.-:%d+:%s*", "")
        return false, "Error: " .. msg, "tool_error"
    end
    if tool.timeout_ms and elapsed_ms > tool.timeout_ms then
        return false, string.format(
            "Error: %s timed out after %.0fms.", name, elapsed_ms), "timeout"
    end
    return true, tostring(result), "continue"
end

--------------------------------------------------------------------------
-- calc: a real expression evaluator.
--
-- The obvious implementation is load("return " .. expr). Do not do that: the
-- expression is model output, and load() hands it the entire Lua runtime.
-- This is the single most common way a toy agent becomes a remote code
-- execution hole, so the repo shows the ~50-line alternative instead.
--
-- Grammar:  expr := term (('+' | '-') term)*
--           term := power (('*' | '/' | '%') power)*
--           power := unary ('^' power)?          -- right associative
--           unary := '-'? primary
--           primary := number | '(' expr ')'
--------------------------------------------------------------------------

local function tokenize(s)
    local out, i = {}, 1
    while i <= #s do
        local c = s:sub(i, i)
        if c:match("%s") then
            i = i + 1
        elseif c:match("%d") or (c == "." and s:sub(i + 1, i + 1):match("%d")) then
            local num = s:match("^%d*%.?%d+", i) or s:match("^%d+", i)
            out[#out + 1] = { t = "num", v = tonumber(num) }
            i = i + #num
        elseif ("+-*/%^()"):find(c, 1, true) then
            out[#out + 1] = { t = c }
            i = i + 1
        else
            return nil, string.format(
                "%q is not arithmetic. calc evaluates numbers and + - * / %% ^ only.", s)
        end
    end
    if #out == 0 then
        return nil, "calc needs an arithmetic expression, for example 4871 * 209."
    end
    return out
end

local function evaluate(toks)
    local pos = 1
    local function peek() return toks[pos] and toks[pos].t end
    local expr

    local function primary()
        local tk = toks[pos]
        if not tk then error("unexpected end of expression", 0) end
        if tk.t == "num" then pos = pos + 1 return tk.v end
        if tk.t == "(" then
            pos = pos + 1
            local v = expr()
            if peek() ~= ")" then error("missing closing parenthesis", 0) end
            pos = pos + 1
            return v
        end
        error("unexpected " .. tk.t .. " in expression", 0)
    end

    local function unary()
        if peek() == "-" then pos = pos + 1 return -unary() end
        if peek() == "+" then pos = pos + 1 return unary() end
        return primary()
    end

    local function power()
        local base = unary()
        if peek() == "^" then
            pos = pos + 1
            return base ^ power()  -- right associative
        end
        return base
    end

    local function term()
        local v = power()
        while peek() == "*" or peek() == "/" or peek() == "%" do
            local op = peek()
            pos = pos + 1
            local rhs = power()
            if (op == "/" or op == "%") and rhs == 0 then
                error("division by zero", 0)
            end
            if op == "*" then v = v * rhs
            elseif op == "/" then v = v / rhs
            else v = v % rhs end
        end
        return v
    end

    expr = function()
        local v = term()
        while peek() == "+" or peek() == "-" do
            local op = peek()
            pos = pos + 1
            if op == "+" then v = v + term() else v = v - term() end
        end
        return v
    end

    local result = expr()
    if pos <= #toks then
        error("unexpected " .. tostring(toks[pos].t) .. " after the expression", 0)
    end
    return result
end

-- Format without float noise: 1018039, not 1018039.0
local function format_number(n)
    if n ~= n then return "nan" end
    if n == math.huge or n == -math.huge then return tostring(n) end
    if n % 1 == 0 and math.abs(n) < 1e15 then return string.format("%d", n) end
    return (string.format("%.10g", n))
end

tools.calc = {
    name = "calc",
    description = "Evaluate an arithmetic expression.",
    args = { { name = "expr", type = "string", required = true } },
    run = function(args)
        local toks, err = tokenize(args.expr)
        if not toks then error(err, 0) end
        return format_number(evaluate(toks))
    end,
}

--------------------------------------------------------------------------
-- Filesystem tools
--------------------------------------------------------------------------

-- Observations are charged against the context budget, so a tool that can
-- return a whole file truncates itself. context.lua handles the window; this
-- stops one pathological read from blowing it in a single step.
tools.MAX_OBSERVATION = 4000

local function truncate(s)
    if #s <= tools.MAX_OBSERVATION then return s end
    return s:sub(1, tools.MAX_OBSERVATION) .. string.format(
        "\n... [truncated, %d bytes total]", #s)
end

tools.read_file = {
    name = "read_file",
    description = "Read a text file.",
    args = { { name = "path", type = "string", required = true } },
    run = function(args)
        local fh, err = io.open(args.path, "r")
        if not fh then
            -- Name the failure precisely: a model that is told the path does
            -- not exist can correct it; one told "error" retries the same path.
            error(string.format("cannot open %q (%s)", args.path,
                tostring(err):gsub("^.*: ", "")), 0)
        end
        local content = fh:read("a")
        fh:close()
        return truncate(content)
    end,
}

tools.list_dir = {
    name = "list_dir",
    description = "List the entries of a directory.",
    args = { { name = "path", type = "string", required = true } },
    run = function(args)
        -- No POSIX module, so shell out. Quoting matters: the path is model
        -- output. Reject anything that could escape the argument.
        if args.path:match('[`$;|&<>"\n]') then
            error("path contains characters that are not allowed", 0)
        end
        local cmd = package.config:sub(1, 1) == "\\"
            and string.format('dir /b "%s" 2>nul', args.path)
            or string.format("ls -1 '%s' 2>/dev/null", args.path)
        local pipe = io.popen(cmd, "r")
        if not pipe then error("cannot list " .. args.path, 0) end
        local out = pipe:read("a")
        pipe:close()
        if not out or out == "" then
            error(string.format("%q is empty or does not exist", args.path), 0)
        end
        return truncate(out)
    end,
}

tools.write_file = {
    name = "write_file",
    description = "Write text to a file, replacing it.",
    args = { { name = "path", type = "string", required = true },
             { name = "text", type = "string", required = true } },
    requires_approval = true,
    run = function(args)
        local fh, err = io.open(args.path, "w")
        if not fh then error(tostring(err), 0) end
        fh:write(args.text)
        fh:close()
        return string.format("wrote %d bytes to %s", #args.text, args.path)
    end,
}

-- The default set, in catalogue order.
function tools.defaults()
    local reg = tools.registry()
    reg:add(tools.calc)
    reg:add(tools.read_file)
    reg:add(tools.list_dir)
    reg:add(tools.write_file)
    return reg
end

return tools
