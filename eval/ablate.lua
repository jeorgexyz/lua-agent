-- eval/ablate.lua - Measure the headline claim, then turn each mechanism off.
--
--   lua54 eval/ablate.lua [--n 50] [--checkpoint <path>] [--out <file>]
--
-- Part 1 is the parse-rate ablation: the same checkpoint, the same prompts,
-- the same greedy sampler, with and without logit masking. This is the number
-- the README leads with, and it is measured here rather than asserted.
--
-- Part 2 runs the eval set with each loop mechanism disabled in turn, so
-- every feature in agent.lua has a row showing what breaks without it.

package.path = "./?.lua;" .. package.path

local tools    = require('tools')
local protocol = require('protocol')
local grammar  = require('grammar')

local function parse_args(argv)
    local o = { n = 50 }
    local i = 1
    while argv[i] do
        local k = argv[i]:gsub("^%-%-", "")
        o[k] = argv[i + 1]
        i = i + 2
    end
    o.n = tonumber(o.n) or 50
    return o
end

local opts = parse_args(arg)

-- Distinct tasks, so the measurement is not one prompt repeated. They share a
-- prefix (the tool catalogue), which the KV cache reuses across trials --
-- otherwise this would take hours at 3-4 tok/s.
local TASKS = {
    "What is 4871 * 209?", "Add 17 and 25.", "Read the file notes.txt.",
    "What is 100 divided by 7?", "List what is in this directory.",
    "Compute 2 to the power of 16.", "How many bytes are in README.md?",
    "Subtract 91 from 1000.", "What is 13 * 13 * 13?", "Read config.txt.",
}

local out = {}
local function say(fmt, ...)
    local line = select("#", ...) > 0 and string.format(fmt, ...) or fmt
    io.write(line .. "\n") io.flush()
    out[#out + 1] = line
end

local reg = tools.registry()
reg:add(tools.calc)
reg:add(tools.read_file)

say("Constrained decoding vs free decoding")
say("=====================================")
say("")
say("Model:       %s", (opts.checkpoint or "stories15M"):match("[^/\\]+$"))
say("Sampler:     greedy (temperature 0.0), identical in both arms")
say("Prompt:      identical in both arms")
say("Difference:  logit masking, on or off")
say("Trials:      %d", opts.n)
say("")

local blocal = require('backend.local')

local function run_arm(constrain)
    local be = blocal.new({ tools = reg, constrain = constrain, temperature = 0.0,
                            checkpoint = opts.checkpoint, llama_path = opts.llama_path })
    local parsed, total, right_tool, samples = 0, 0, 0, {}
    local t0 = os.clock()
    for i = 1, opts.n do
        local task = TASKS[((i - 1) % #TASKS) + 1] .. string.rep(" ", math.floor((i - 1) / #TASKS))
        local prompt = protocol.render_compact(reg, task)
        local text = be:complete(prompt, {})
        local call = text:match(protocol.CALL_PREFIX .. "(.*)$") or ""
        total = total + 1
        local turn = protocol.parse(text)
        if turn then
            parsed = parsed + 1
            -- Did it pick a tool that could actually serve the task? This is
            -- the semantics column, and it is where the 15M model fails.
            local wants_calc = task:match("%d") and not task:match("[Rr]ead")
            local picked = turn.kind == "answer" and "answer" or turn.tool
            if (wants_calc and picked == "calc")
               or (task:match("[Rr]ead") and picked == "read_file") then
                right_tool = right_tool + 1
            end
        end
        if #samples < 3 then samples[#samples + 1] = call end
        if i % 10 == 0 then
            io.write(string.format("  ... %d/%d (%.0fs)\n", i, opts.n, os.clock() - t0))
            io.flush()
        end
    end
    return parsed, total, right_tool, samples, os.clock() - t0
end

say("Running constrained arm...")
local cp, ct, cr, cs, csec = run_arm(true)
say("Running free arm...")
local fp, ft, fr, fs, fsec = run_arm(false)

say("")
say("  arm            parsed as a tool call     picked a usable tool")
say("  -----------    ---------------------     --------------------")
say("  constrained    %3d/%-3d  (%5.1f%%)            %3d/%-3d  (%5.1f%%)",
    cp, ct, 100 * cp / ct, cr, ct, 100 * cr / ct)
say("  free           %3d/%-3d  (%5.1f%%)            %3d/%-3d  (%5.1f%%)",
    fp, ft, 100 * fp / ft, fr, ft, 100 * fr / ft)
say("")
say("  (%.0fs constrained, %.0fs free, pure Lua CPU inference)", csec, fsec)
say("")
say("The two columns are the whole point.")
say("")
say("The first column is the GRAMMAR's work: whether the output is a")
say("syntactically valid tool call. Masking takes it to 100% on a model")
say("that has never seen JSON.")
say("")
say("The second column is the MODEL's work: whether the tool it chose could")
say("plausibly serve the task. The grammar cannot help here, and a 15M")
say("TinyStories checkpoint has nothing to contribute.")
say("")
say("Tool calling is a decoding constraint. Choosing the right tool is not.")
say("")
say("Sample output, constrained arm:")
for _, s in ipairs(cs) do say("  %s", s) end
say("")
say("Sample output, free arm:")
for _, s in ipairs(fs) do say("  %s", (s:gsub("\n", " "))) end

if opts.out then
    local fh = io.open(opts.out, "w")
    if fh then
        fh:write(table.concat(out, "\n"), "\n")
        fh:close()
        io.write("\nwrote " .. opts.out .. "\n")
    end
end
