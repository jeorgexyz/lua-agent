-- eval/run.lua - Run the eval set and ablate each loop mechanism.
--
--   lua54 eval/run.lua                    baseline, scripted model
--   lua54 eval/run.lua --ablate           every mechanism turned off in turn
--   lua54 eval/run.lua --backend local    the same tasks against stories15M
--   lua54 eval/run.lua --backend ollama   against a local instruct model
--   lua54 eval/run.lua --out examples/eval.txt
--
-- Run from the repository root; the tasks reference eval/fixtures by
-- relative path.
--
-- The default model is SCRIPTED -- a fixed list of turns per task, described
-- in eval/tasks/tasks.lua. That is the point: an ablation of the loop needs
-- the model held constant, or you cannot tell whether removing the loop
-- detector changed the outcome or the model just sampled differently.
--
-- Every feature in agent.lua earns its place by a row in the ablation table
-- showing what breaks without it.

package.path = "./?.lua;" .. package.path

local Agent    = require('agent')
local tools    = require('tools')
local protocol = require('protocol')
local TASKS = require('eval.tasks.tasks').tasks

local function parse_args(argv)
    local o, i = {}, 1
    while argv[i] do
        local k = argv[i]:gsub("^%-%-", "")
        if k == "ablate" or k == "verbose" then
            o[k] = true i = i + 1
        else
            o[k] = argv[i + 1] i = i + 2
        end
    end
    return o
end

local opts = parse_args(arg)
local out_lines = {}
local function say(fmt, ...)
    local s = select("#", ...) > 0 and string.format(fmt, ...) or fmt
    io.write(s .. "\n") io.flush()
    out_lines[#out_lines + 1] = s
end

-- A fixed list of turns, replayed in order. Not a model.
local function scripted(turns)
    local i = 0
    return { complete = function()
        i = i + 1
        return turns[i] or
            'THOUGHT: out of script\nCALL: {"tool":"answer","args":{"text":"(script exhausted)"}}',
            {}
    end }
end

-- Approximate tokenizer: the loop ablation is about control flow, and
-- loading a 60MB checkpoint to count tokens for it would be silly. The
-- oversized-observation task sets its budget in these same units, so the
-- comparison is internally consistent.
local approx_tokenizer = { encode = function(_, s)
    local out = {}
    for k = 1, math.max(1, math.ceil(#s / 4)) do out[k] = k end
    return out
end }

local function build_registry(names)
    local reg = tools.registry()
    for _, n in ipairs(names) do reg:add(tools[n]) end
    return reg
end

-- A real backend, loaded once and reused across tasks -- a checkpoint load
-- is ~17s and there are seven tasks. nil means the scripted model.
local REAL = nil
if opts.backend == "local" or opts.backend == "http"
   or opts.backend == "ollama" then
    -- The compact catalogue: stories15M's whole context is 256 tokens and
    -- the verbose one costs 194 for two tools.
    local real_render = protocol.render_system
    protocol.render_system = function(t, task) return real_render(t, task, true) end

    if opts.backend == "local" then
        REAL = require('backend.local').new({
            llama_path = opts.llama_path,
            checkpoint = opts.checkpoint,
            tokenizer = opts.tokenizer,
            -- Compiled per task below; this is just to satisfy the
            -- constructor with a non-empty registry.
            tools = build_registry({ "calc" }),
            constrain = opts.no_constrain == nil,
            temperature = 0.0,
        })
    elseif opts.backend == "ollama" then
        REAL = require('backend.ollama').new({
            model = opts.model, host = opts.ollama_host,
            -- Replaced per task by set_tools below; the constructor just
            -- needs a non-empty registry to build a schema from.
            tools = build_registry({ "calc" }),
            format = opts.ollama_free == nil,
            temperature = 0,
        })
    else
        REAL = require('backend.http').new({ model = opts.model, temperature = 0 })
    end
end

-- variant: { repeat_threshold=, max_tokens=, policy=, fatal_on= }
local function run_task(task, variant)
    variant = variant or {}
    local reg = build_registry(task.tools)

    local backend = scripted(task.script)
    local tokenizer = approx_tokenizer
    if REAL then
        backend = REAL
        if REAL.set_tools then REAL:set_tools(reg) end
        if REAL.reset then REAL:reset() end
        -- Real token counting when the backend brought a real tokenizer.
        if REAL.tok then tokenizer = REAL.tok end
    end

    local agent = Agent.new({
        backend = backend,
        tools = reg,
        tokenizer = tokenizer,
        -- 4 was chosen when a step cost ~2 minutes on the pure-Lua backend.
        -- Under Ollama a step is seconds, and a ceiling that tight scores a
        -- model as failing when it was still making progress -- three tasks
        -- ended on read_file,read_file,read_file,read_file with no room left
        -- to answer.
        max_steps = variant.max_steps or (REAL and 8 or 10),
        max_tokens = variant.max_tokens or task.budget
            or (REAL and REAL.cfg and math.floor(REAL.cfg.seq_len * 0.75)) or 4096,
        repeat_threshold = variant.repeat_threshold or 3,
        policy = variant.policy,
        fatal_on = variant.fatal_on,
        approve = function() return task.approve == true end,
    })

    local answer, reason, steps = agent:run(task.prompt)

    local evicted, tokens = false, 0
    for _, s in ipairs(steps) do
        if s.evicted then evicted = true end
        if s.tokens and s.tokens.window then tokens = math.max(tokens, s.tokens.window) end
    end

    -- Which tools the model actually reached for. With a real checkpoint the
    -- pass/fail column alone is opaque -- "expected a parse error to occur"
    -- is technically true and tells you nothing. What you want to know is
    -- that the model answered immediately and never touched a tool.
    local chose = {}
    for _, s in ipairs(steps) do
        chose[#chose + 1] = s.call and s.call.tool or (s.status or "?")
    end

    -- Against the scripted model, ask whether the mechanism this task exists
    -- for actually fired. Against a real one, ask whether the task got done.
    -- See the header of eval/tasks/tasks.lua for why those are different
    -- questions -- conflating them scored a model as failing for not making
    -- the mistake the script was written to make.
    local judge = (REAL and task.solved) or task.check
    local ok, why = judge(answer, steps, { evicted = evicted, reason = reason })
    return ok, why, #steps, tokens, reason, table.concat(chose, ",")
end

local function run_suite(variant, label)
    local passed, total, notes = 0, 0, {}
    for _, task in ipairs(TASKS) do
        total = total + 1
        local ok, why = false, nil
        local safe, err = pcall(function()
            ok, why = run_task(task, variant)
        end)
        if not safe then ok, why = false, "crashed: " .. tostring(err):gsub("^.-:%d+:%s*", "") end
        if ok then
            passed = passed + 1
        else
            notes[#notes + 1] = task.id .. ": " .. tostring(why)
        end
    end
    return passed, total, notes
end

--------------------------------------------------------------------------
-- Baseline
--------------------------------------------------------------------------

say("Agent loop evaluation")
say("=====================")
say("")
if REAL then
    local name = opts.model or opts.checkpoint or "../lua_llama/stories15M.bin"
    say("Model:  %s  (%s backend, real inference)",
        name:match("[^/\\]+$") or name, opts.backend)
    if REAL.cfg then
        say("Config: dim=%d layers=%d seq_len=%d  (window budget %d)",
            REAL.cfg.dim, REAL.cfg.n_layers, REAL.cfg.seq_len,
            math.floor(REAL.cfg.seq_len * 0.75))
    end
    -- Only the local backend runs grammar.lua. Ollama and the API own their
    -- decoders, so a pass there is evidence about the loop and the model --
    -- not about constrained decoding. Saying which is which here keeps the
    -- two results from being read as one.
    if opts.backend == "local" then
        say("Decode: %s (grammar.lua)", REAL.constrain and "constrained" or "free")
    else
        say("Decode: the backend's own (grammar.lua is not in this loop)")
    end
else
    say("Model:  scripted (a fixed turn list per task, see eval/tasks/tasks.lua)")
    say("Why:    an ablation of the LOOP needs the model held constant.")
    say("        --backend local runs the same tasks against a real checkpoint.")
end
say("")
if REAL then
    say("  %-24s %-6s %-6s %-8s %s", "task", "pass", "steps", "window", "tools it chose")
    say("  %-24s %-6s %-6s %-8s %s", string.rep("-", 24), "----", "-----", "------",
        string.rep("-", 24))
else
    say("  %-26s %-6s %-6s %s", "task", "pass", "steps", "window")
    say("  %-26s %-6s %-6s %s", string.rep("-", 26), "----", "-----", "------")
end

local passed, total = 0, 0
for _, task in ipairs(TASKS) do
    total = total + 1
    local ok, why, steps, tokens, _, chose = run_task(task, nil)
    if ok then passed = passed + 1 end
    if REAL then
        say("  %-24s %-6s %-6d %-8d %s", task.id, ok and "ok" or "FAIL",
            steps, tokens, chose)
    else
        say("  %-26s %-6s %-6d %d%s", task.id, ok and "ok" or "FAIL", steps, tokens,
            ok and "" or ("   <- " .. tostring(why)))
    end
end
say("")
say("  %d/%d passed", passed, total)

--------------------------------------------------------------------------
-- Ablations
--------------------------------------------------------------------------

if opts.ablate then
    say("")
    say("Ablations")
    say("---------")
    say("Each row disables one mechanism. The tasks and the scripted model")
    say("are identical in every row; only the loop changes.")
    say("")

    local variants = {
        { "(baseline)", {}, "" },
        { "no loop detector", { repeat_threshold = 10000 },
          "loop-bait spins to the step ceiling" },
        { "no parse recovery", { fatal_on = { parse_error = true } },
          "a truncated JSON call kills the run" },
        { "no tool-error recovery", { fatal_on = { tool_error = true } },
          "one bad path kills the run" },
        { "no approval gate", { }, "(gate is not optional; see note below)" },
        { "no context eviction", { max_tokens = 100000 },
          "oversized observation never triggers eviction" },
        { "policy=drop_oldest", { policy = "drop_oldest" },
          "drops whole steps instead of eliding results" },
    }

    say("  %-26s %-7s %s", "ablation", "pass", "what breaks")
    say("  %-26s %-7s %s", string.rep("-", 26), "-----", string.rep("-", 40))
    for _, v in ipairs(variants) do
        local name, variant = v[1], v[2]
        if name == "no approval gate" then
            say("  %-26s %-7s %s", name, "n/a",
                "the gate is default-deny in tools.lua, not a loop flag")
        else
            local p, t, notes = run_suite(variant, name)
            local first = notes[1] and notes[1]:gsub(":.*", "") or ""
            say("  %-26s %d/%-5d %s", name, p, t,
                p == t and "nothing" or (first .. " fails"))
        end
    end
end

if opts.out then
    local fh = io.open(opts.out, "w")
    if fh then fh:write(table.concat(out_lines, "\n"), "\n") fh:close()
        io.write("\nwrote " .. opts.out .. "\n") end
end

os.exit(passed == total and 0 or 1)
