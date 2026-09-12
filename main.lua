-- main.lua - CLI entry point.
--
--   lua54 main.lua "<task>" [options]
--
-- Options:
--   --backend local|ollama|http|replay   default local
--   --llama-path <dir>            lua-llama checkout, default ../lua_llama
--   --checkpoint <path>           default <llama-path>/stories15M.bin
--   --tokenizer <path>            default <llama-path>/tokenizer.bin
--   --no-constrain                free decoding; the 0% arm of the ablation
--   --tools a,b,c                 default calc,read_file
--   --mcp "<command>"             mount an MCP server's tools over stdio
--   --mcp-prefix <str>            namespace them, default "mcp_"
--   --mcp-trust                   let MCP tools run without approval
--   --max-steps <n>               default 6
--   --max-tokens <n>              context budget; defaults to the model's window
--   --policy <name>               drop_oldest|elide_observations|summarize
--   --full-prompt                 verbose tool catalogue (needs a big window)
--   --temperature <t>             default 0.0
--   --trace <path>                record a JSONL trace
--   --replay <path>               play a recorded trace back
--   --model <name>                http/ollama model name
--   --ollama-host <url>           default http://localhost:11434
--   --ollama-free                 turn OFF Ollama's schema (free decoding)
--   --yes                         auto-approve gated tools
--   --quiet                       suppress per-step output

package.path = "./?.lua;" .. package.path

local Agent    = require('agent')
local tools    = require('tools')
local protocol = require('protocol')

local FLAGS = {
    ["--no-constrain"] = true, ["--yes"] = true, ["--quiet"] = true,
    ["--full-prompt"] = true, ["--help"] = true, ["-h"] = true,
    ["--ollama-free"] = true, ["--mcp-trust"] = true,
}

local function parse_args(argv)
    local o = { task = nil }
    local i = 1
    while argv[i] do
        local a = argv[i]
        if a:sub(1, 2) == "--" or a == "-h" then
            local key = a:gsub("^%-%-?", ""):gsub("%-", "_")
            if FLAGS[a] then
                o[key] = true
                i = i + 1
            else
                local v = argv[i + 1]
                if v == nil then
                    io.stderr:write("missing value for " .. a .. "\n")
                    os.exit(2)
                end
                o[key] = v
                i = i + 2
            end
        else
            o.task = o.task and (o.task .. " " .. a) or a
            i = i + 1
        end
    end
    return o
end

local function build_tools(spec)
    local reg = tools.registry()
    for name in (spec or "calc,read_file"):gmatch("[^,]+") do
        name = name:match("^%s*(.-)%s*$")
        local t = tools[name]
        if not t then
            io.stderr:write("unknown tool: " .. name .. "\n")
            os.exit(2)
        end
        reg:add(t)
    end
    return reg
end

local function build_backend(o, reg)
    if o.replay then
        return require('backend.replay').new({ path = o.replay })
    elseif o.backend == "http" then
        return require('backend.http').new({
            model = o.model,
            temperature = tonumber(o.temperature) or 0,
        })
    elseif o.backend == "ollama" then
        return require('backend.ollama').new({
            model = o.model,
            host = o.ollama_host,
            tools = reg,
            -- Schema-constrained unless --ollama-free; see backend/ollama.lua.
            format = not o.ollama_free,
            temperature = tonumber(o.temperature) or 0,
        })
    else
        return require('backend.local').new({
            llama_path = o.llama_path,
            checkpoint = o.checkpoint,
            tokenizer = o.tokenizer,
            tools = reg,
            constrain = not o.no_constrain,
            temperature = tonumber(o.temperature) or 0.0,
        })
    end
end

-- The approval gate. Read-only tools never reach here; write and exec tools
-- always do. Defaults to no -- an agent that writes to disk because the
-- prompt was ambiguous is the failure this exists to prevent.
local function make_approver(auto)
    if auto then
        return function(call)
            io.write("  [auto-approved: " .. call.tool .. "]\n")
            return true
        end
    end
    return function(call)
        io.write(string.format("\n  APPROVE %s %s? [y/N] ", call.tool,
            require('trace').encode(call.args)))
        io.flush()
        local line = io.read("l")
        return line ~= nil and line:lower():sub(1, 1) == "y"
    end
end

local function main(argv)
    local o = parse_args(argv)

    if o.help or o.h or not o.task then
        local src = io.open(arg[0] or "main.lua", "r")
        if src then
            for line in src:lines() do
                if line:sub(1, 2) ~= "--" then break end
                io.write(line:sub(4) .. "\n")
            end
            src:close()
        end
        os.exit(o.task and 0 or 2)
    end

    local reg = build_tools(o.tools)

    -- An MCP server's tools join the registry alongside the built-ins and
    -- are indistinguishable to the loop from there: same validation, same
    -- approval gate, same grammar. Mounted BEFORE the backend is built,
    -- because the grammar and the Ollama schema are compiled from the
    -- registry and have to include them.
    if o.mcp then
        local mcp = require('mcp')
        local client = mcp.new({ command = o.mcp })
        local ok, err = pcall(function()
            mcp.mount(reg, client, {
                prefix = o.mcp_prefix or "mcp_",
                trust = o.mcp_trust,
            })
        end)
        if not ok then
            io.stderr:write("MCP: " .. tostring(err) .. "\n")
            os.exit(2)
        end
        io.write(string.format("mcp:     %s (%s) -- %d tools\n",
            client.server_info and client.server_info.name or "?",
            client.negotiated_version or "?", #(client.tools or {})))
    end

    local backend = build_backend(o, reg)

    -- Token counting needs a tokenizer, but NOT a model. The local backend
    -- already has one loaded; every other backend loads the 433KB
    -- tokenizer.bin on its own rather than falling back to guesswork.
    local bpe = require('bpe')
    local tokenizer = backend.tok
    local approximate, tok_note = false, nil

    if not tokenizer then
        local path = o.tokenizer
            or ((o.llama_path or "../lua_llama") .. "/tokenizer.bin")
        local loaded, err = bpe.load(path, o.llama_path)
        if loaded then
            tokenizer = loaded
        else
            approximate = true
            tokenizer = bpe.approximate()
            tok_note = err
        end
    end

    -- The context budget belongs to the MODEL, not the tokenizer. This used
    -- to key off "do we have a real tokenizer", which happened to be true
    -- only for the local backend -- so once the tokenizer could load on its
    -- own, a replay run silently inherited stories15M's 256-token window.
    local default_budget = 4096
    if backend.cfg and backend.cfg.seq_len then
        default_budget = math.floor(backend.cfg.seq_len * 0.75)
    end

    -- The verbose catalogue costs 194 tokens for two tools; stories15M's
    -- whole context is 256. Compact by default, opt out with --full-prompt.
    local compact = not o.full_prompt
    local real_render = protocol.render_system
    if compact then
        protocol.render_system = function(t, task) return real_render(t, task, true) end
    end

    local agent = Agent.new({
        backend = backend,
        tools = reg,
        tokenizer = tokenizer,
        max_steps = tonumber(o.max_steps) or 6,
        max_tokens = tonumber(o.max_tokens) or default_budget,
        policy = o.policy,
        approve = make_approver(o.yes),
        trace_path = o.trace,
        verbose = not o.quiet,
    })

    io.write(string.format("task:    %s\nbackend: %s%s\ntools:   %s\n",
        o.task,
        o.replay and ("replay " .. o.replay) or (o.backend or "local"),
        (not o.replay and (o.backend == nil or o.backend == "local"))
            and (o.no_constrain and " (free decoding)" or " (grammar-constrained)") or "",
        table.concat(reg:names(), ", ")))
    if approximate then
        -- Say what it is and how to fix it, in one line. This shows up in
        -- every transcript, so it has to be informative without reading as
        -- a defect.
        io.write("         token counts estimated -- pass --tokenizer <tokenizer.bin> for exact\n")
        if o.verbose_tokenizer then
            io.write("         (" .. tostring(tok_note) .. ")\n")
        end
    end

    local started = os.clock()
    local answer, reason, steps = agent:run(o.task)
    local elapsed = os.clock() - started

    io.write("\n" .. string.rep("-", 60) .. "\n")
    if answer then
        io.write("ANSWER: " .. answer .. "\n")
    else
        io.write("No answer. Stopped because: " .. reason .. "\n")
    end

    local parsed = 0
    for _, s in ipairs(steps) do
        if s.call then parsed = parsed + 1 end
    end
    io.write(string.format("%d steps, %d/%d turns parsed as a tool call, %.1fs\n",
        #steps, parsed, #steps, elapsed))
    if o.trace then io.write("trace:  " .. o.trace .. "\n") end

    os.exit(answer and 0 or 1)
end

main(arg)
