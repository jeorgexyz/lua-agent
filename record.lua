-- record.lua - Regenerate traces/ and examples/ from the eval set.
--
--   lua54 record.lua
--
-- Writes, for each eval task:
--   traces/<id>.jsonl    the machine-readable trace, replayable
--   examples/<id>.txt    the human-readable transcript
--
-- The recorded prompts must match what main.lua builds by default, or
-- replaying needs a pile of flags to line the prompt back up. So this uses
-- compact rendering (main.lua's default) and the task's own tool set, and
-- writes the exact replay command into each transcript.

package.path = "./?.lua;" .. package.path

local Agent    = require('agent')
local tools    = require('tools')
local trace    = require('trace')
local protocol = require('protocol')
local TASKS    = require("eval.tasks.tasks").tasks

-- One place, so the recording and the printed replay command cannot drift.
local MAX_STEPS = 10

-- Match main.lua's default: the compact catalogue.
local real_render = protocol.render_system
protocol.render_system = function(t, task) return real_render(t, task, true) end

local function scripted(turns)
    local i = 0
    return { complete = function() i = i + 1 return turns[i], {} end }
end

-- Record with the real tokenizer when it is reachable, so the transcripts
-- and the demo GIFs show the same token counts a reader will see when they
-- run the replay command. Falls back to the estimator so this still works
-- in a checkout without repo 1 beside it.
local bpe = require('bpe')
local tokenizer, vocab = bpe.load("../lua_llama/tokenizer.bin")
if not tokenizer then
    tokenizer = bpe.approximate()
    io.write("note: recording with estimated token counts (", tostring(vocab), ")\n")
end

for _, task in ipairs(TASKS) do
    local reg = tools.registry()
    for _, n in ipairs(task.tools) do reg:add(tools[n]) end

    local agent = Agent.new({
        backend = scripted(task.script),
        tools = reg,
        tokenizer = tokenizer,
        max_steps = MAX_STEPS,
        max_tokens = task.budget or 4096,
        approve = function() return task.approve == true end,
        trace_path = "traces/" .. task.id .. ".jsonl",
    })

    local answer, reason, steps = agent:run(task.prompt)

    -- The printed command must reproduce the recording exactly, which means
    -- naming every setting where main.lua's default differs from the one
    -- used here. --max-steps is the one that bit: main.lua defaults to 6,
    -- loop-bait needs 7, and the replay silently stopped one step short of
    -- the answer -- reported as a max_steps failure of a run that had in
    -- fact succeeded.
    local cmd = string.format(
        'lua54 main.lua "%s" --replay traces/%s.jsonl --tools %s --max-steps %d%s%s',
        task.prompt, task.id, table.concat(task.tools, ","), MAX_STEPS,
        task.budget and (" --max-tokens " .. task.budget) or "",
        -- Without --yes the gate prompts on stdin and defaults to deny, so
        -- a replay of an approved run would silently become a denied one.
        task.approve and " --yes" or "")

    local out = { "task:   " .. task.prompt,
                  "tools:  " .. table.concat(task.tools, ", "),
                  "model:  scripted (see eval/tasks/tasks.lua)",
                  "replay: " .. cmd, "" }
    for _, s in ipairs(steps) do
        out[#out + 1] = string.format("[step %d] %s", s.step, s.status)
        if s.tokens and s.tokens.window and s.tokens.window > 0 then
            out[#out + 1] = string.format("  window: %d / %d tokens%s",
                s.tokens.window, s.tokens.budget, s.evicted and "  (evicted)" or "")
        end
        if s.thought then out[#out + 1] = "  thought: " .. s.thought end
        if s.call then
            out[#out + 1] = string.format('  call:    {"tool":"%s","args":%s}',
                s.call.tool, trace.encode(s.call.args or {}))
        end
        if s.observation then
            local o = s.observation
            if #o > 300 then o = o:sub(1, 300) .. "\n... [trimmed for display]" end
            out[#out + 1] = "  result:  " .. o:gsub("\n", "\n           ")
        end
        out[#out + 1] = ""
    end
    out[#out + 1] = string.rep("-", 60)
    out[#out + 1] = answer and ("ANSWER: " .. answer)
                            or ("No answer. Stopped because: " .. reason)

    local fh = io.open("examples/" .. task.id .. ".txt", "w")
    fh:write(table.concat(out, "\n"), "\n")
    fh:close()

    print(string.format("%-24s %-8s %s", task.id, reason, tostring(answer)))
end
