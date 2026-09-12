-- eval/tasks/tasks.lua - The eval set.
--
-- Repo 1's falsifiable claim is "output matches llama2.c". This is repo 2's:
-- tasks with checkable answers, where `check` also sees the step list -- so
-- producing the right answer WITHOUT calling the tool still fails.
--
--
-- WHY THE DEFAULT MODEL HERE IS SCRIPTED
--
-- Each task ships a `script`: a list of model turns, replayed in order. That
-- is a simulated model, and it is labelled as one everywhere it is reported.
-- It is the default because the thing being measured is the LOOP, and a loop
-- ablation needs the model held fixed. With a real model you cannot tell
-- whether removing the loop detector changed the outcome or whether the
-- model simply sampled differently that run.
--
-- The scripts are not strawmen. Each encodes a specific mistake real models
-- actually make: a malformed call, a wrong path, a tool that keeps returning
-- the same unhelpful thing, an oversized read. The loop's job is to survive
-- them. `--backend local|http` runs the same tasks against a real model.

-- TWO KINDS OF ASSERTION, KEPT APART
--
-- `check` asks "did the loop exercise the mechanism this task exists for" --
-- a parse error occurred and was recovered from, the loop detector fired,
-- eviction happened. That only makes sense against the scripted model,
-- which is built to trigger it.
--
-- `solved` asks "is the answer right, and did it come from the tools" --
-- which is the question for a real model.
--
-- They were one function until qwen2.5:1.5b ran the suite and "failed"
-- recover-parse-error by never making a parse error (the schema prevents
-- them) and loop-bait by never looping (it read the file, found nothing,
-- and said so). Both were the model outperforming the script, scored as
-- failures. A task that demands a specific failure mode occur is testing
-- the harness, not the agent -- so the two questions are now separate and
-- eval/run.lua asks whichever one fits the backend.

local M = {}

-- Did the agent actually call this tool, rather than guessing the answer?
local function used(steps, tool)
    for _, s in ipairs(steps) do
        if s.call and s.call.tool == tool then return true end
    end
    return false
end

local function saw_status(steps, status)
    for _, s in ipairs(steps) do
        if s.status == status then return true end
    end
    return false
end

local function contains(answer, want)
    return answer ~= nil and tostring(answer):find(want, 1, true) ~= nil
end

M.tasks = {
    --------------------------------------------------------------------
    {
        id = "calc-single",
        prompt = "What is 4871 * 209?",
        tools = { "calc" },
        -- The plain case: one tool, one answer. Everything else is a
        -- variation on this with something going wrong.
        script = {
            'THOUGHT: I cannot do this in my head.\nCALL: {"tool":"calc","args":{"expr":"4871 * 209"}}',
            'THOUGHT: I have the product.\nCALL: {"tool":"answer","args":{"text":"1018039"}}',
        },
        solved = function(answer, steps)
            if not used(steps, "calc") then return false, "answered without calling calc" end
            if not contains(answer, "1018039") then
                return false, "wrong product: " .. tostring(answer)
            end
            return true
        end,
        check = function(answer, steps)
            if not used(steps, "calc") then
                return false, "answered without calling calc"
            end
            if not contains(answer, "1018039") then
                return false, "wrong product: " .. tostring(answer)
            end
            return true
        end,
    },

    --------------------------------------------------------------------
    {
        id = "read-then-calc",
        prompt = "How many lines are in eval/fixtures/notes.txt, times 3?",
        tools = { "read_file", "calc" },
        -- Multi-hop: the output of one tool is the input to the next.
        script = {
            'THOUGHT: First read the file.\nCALL: {"tool":"read_file","args":{"path":"eval/fixtures/notes.txt"}}',
            'THOUGHT: Four lines, so four times three.\nCALL: {"tool":"calc","args":{"expr":"4 * 3"}}',
            'THOUGHT: Done.\nCALL: {"tool":"answer","args":{"text":"12"}}',
        },
        solved = function(answer, steps)
            if not (used(steps, "read_file") and used(steps, "calc")) then
                return false, "did not chain both tools"
            end
            if not contains(answer, "12") then return false, "wrong result: " .. tostring(answer) end
            return true
        end,
        check = function(answer, steps)
            if not (used(steps, "read_file") and used(steps, "calc")) then
                return false, "did not chain both tools"
            end
            if not contains(answer, "12") then
                return false, "wrong result: " .. tostring(answer)
            end
            return true
        end,
    },

    --------------------------------------------------------------------
    {
        id = "recover-parse-error",
        prompt = "What is 2 + 2?",
        tools = { "calc" },
        -- The model emits a truncated call. Without parse recovery the run
        -- dies here; with it, the parser's message goes back as the
        -- observation and the model corrects itself.
        script = {
            'THOUGHT: Compute it.\nCALL: {"tool":"calc","args":{"expr":"2 + 2"',
            'THOUGHT: My JSON was cut off, let me redo it.\nCALL: {"tool":"calc","args":{"expr":"2 + 2"}}',
            'THOUGHT: Four.\nCALL: {"tool":"answer","args":{"text":"4"}}',
        },
        -- A schema-constrained backend cannot emit a malformed call at all,
        -- so "a parse error happened" is not something to require of it.
        solved = function(answer, steps)
            if not used(steps, "calc") then return false, "answered without calling calc" end
            if not contains(answer, "4") then return false, "wrong answer: " .. tostring(answer) end
            return true
        end,
        check = function(answer, steps)
            if not saw_status(steps, "parse_error") then
                return false, "expected a parse error to occur"
            end
            if not contains(answer, "4") then
                return false, "did not recover: " .. tostring(answer)
            end
            return true
        end,
    },

    --------------------------------------------------------------------
    {
        id = "recover-missing-file",
        prompt = "Read the notes file and tell me the first word.",
        tools = { "read_file", "list_dir" },
        -- Wrong path first. The tool error must be specific enough that the
        -- model can correct it -- which is why tools.lua names the path in
        -- the message instead of saying "error".
        script = {
            'THOUGHT: Try the obvious path.\nCALL: {"tool":"read_file","args":{"path":"notes.txt"}}',
            'THOUGHT: Not there. Look in the fixtures directory.\nCALL: {"tool":"read_file","args":{"path":"eval/fixtures/notes.txt"}}',
            'THOUGHT: The first word is alpha.\nCALL: {"tool":"answer","args":{"text":"alpha"}}',
        },
        solved = function(answer, steps)
            if not contains(answer, "alpha") then
                return false, "did not reach the file: " .. tostring(answer)
            end
            return true
        end,
        check = function(answer, steps)
            if not saw_status(steps, "tool_error") then
                return false, "expected the first read to fail"
            end
            if not contains(answer, "alpha") then
                return false, "did not recover: " .. tostring(answer)
            end
            return true
        end,
    },

    --------------------------------------------------------------------
    {
        id = "loop-bait",
        prompt = "Find the secret in the config.",
        tools = { "read_file" },
        -- The file exists and simply does not contain a secret. A model
        -- without a loop detector re-reads it forever. The detector fires,
        -- the nudge lands, and the model gives up honestly.
        script = {
            'THOUGHT: Read the config.\nCALL: {"tool":"read_file","args":{"path":"eval/fixtures/config.txt"}}',
            'THOUGHT: Maybe I missed it.\nCALL: {"tool":"read_file","args":{"path":"eval/fixtures/config.txt"}}',
            'THOUGHT: Check once more.\nCALL: {"tool":"read_file","args":{"path":"eval/fixtures/config.txt"}}',
            'THOUGHT: Once more.\nCALL: {"tool":"read_file","args":{"path":"eval/fixtures/config.txt"}}',
            'THOUGHT: Once more.\nCALL: {"tool":"read_file","args":{"path":"eval/fixtures/config.txt"}}',
            'THOUGHT: Once more.\nCALL: {"tool":"read_file","args":{"path":"eval/fixtures/config.txt"}}',
            'THOUGHT: The nudge is right, there is no secret there.\nCALL: {"tool":"answer","args":{"text":"no secret found in the config"}}',
        },
        -- A model that reads the config once, sees no secret and says so has
        -- done the task correctly. Requiring the loop detector to fire would
        -- punish it for not thrashing.
        solved = function(answer, steps)
            if answer == nil then return false, "never answered" end
            local a = tostring(answer):lower()
            if a:find("no secret") or a:find("not find") or a:find("no ")
               or a:find("none") then return true end
            return false, "did not report the absence: " .. tostring(answer)
        end,
        check = function(answer, steps)
            if not saw_status(steps, "repeat") then
                return false, "loop detector never fired"
            end
            if answer == nil then
                return false, "ran out of steps instead of answering"
            end
            return true
        end,
    },

    --------------------------------------------------------------------
    {
        id = "oversized-observation",
        prompt = "What is the MARKER line at the top of eval/fixtures/big.txt?",
        tools = { "read_file", "calc" },
        -- An observation larger than the budget. Eviction has to fire, and
        -- the model has to be TOLD -- otherwise it invents the contents.
        --
        -- The marker sits at the TOP of the file on purpose. It used to be
        -- the last word, which the scripted model "knew" from its script
        -- and no real model could ever see: tools.MAX_OBSERVATION truncates
        -- the read at 4000 bytes, roughly a fifth of the file. The task was
        -- unsolvable by anything that actually reads, and only qwen2.5
        -- running it revealed that. The observation is still far larger
        -- than the 400-token budget, so eviction fires exactly as before.
        budget = 400,
        script = {
            'THOUGHT: Read the big file.\nCALL: {"tool":"read_file","args":{"path":"eval/fixtures/big.txt"}}',
            'THOUGHT: Add something to push the window.\nCALL: {"tool":"calc","args":{"expr":"1 + 1"}}',
            'THOUGHT: And again.\nCALL: {"tool":"calc","args":{"expr":"2 + 2"}}',
            'THOUGHT: The last word is omega.\nCALL: {"tool":"answer","args":{"text":"omega"}}',
        },
        solved = function(answer, steps)
            if not contains(answer, "omega") then
                return false, "wrong answer: " .. tostring(answer)
            end
            return true
        end,
        check = function(answer, steps, meta)
            if not (meta and meta.evicted) then
                return false, "eviction never fired under a 400-token budget"
            end
            if not contains(answer, "omega") then
                return false, "wrong answer: " .. tostring(answer)
            end
            return true
        end,
    },

    --------------------------------------------------------------------
    {
        id = "approved-write",
        prompt = "Write the number 42 to out.txt, then read it back to confirm.",
        tools = { "write_file", "read_file" },
        approve = true,
        -- The gate's normal path: a write is proposed, approved, and
        -- happens. The denied-write task below covers the refusal, but this
        -- is the one that shows what the gate is FOR -- an agent that can
        -- act on the filesystem, with a checkpoint in front of it.
        script = {
            'THOUGHT: Write the value first.\nCALL: {"tool":"write_file","args":{"path":"out.txt","text":"42"}}',
            'THOUGHT: Read it back to be sure it landed.\nCALL: {"tool":"read_file","args":{"path":"out.txt"}}',
            'THOUGHT: Confirmed.\nCALL: {"tool":"answer","args":{"text":"Wrote 42 to out.txt and read it back."}}',
        },
        solved = function(answer, steps)
            if not used(steps, "write_file") then return false, "never called write_file" end
            local f = io.open("out.txt", "r")
            if not f then return false, "the file was not written" end
            local body = f:read("a") f:close() os.remove("out.txt")
            if not body:find("42", 1, true) then
                return false, "wrong contents: " .. tostring(body)
            end
            return true
        end,
        check = function(answer, steps)
            if not used(steps, "write_file") then
                return false, "never called write_file"
            end
            local f = io.open("out.txt", "r")
            if not f then return false, "the file was not written" end
            local body = f:read("a")
            f:close()
            os.remove("out.txt")
            if not body:find("42", 1, true) then
                return false, "wrong contents: " .. tostring(body)
            end
            if not contains(answer, "42") then
                return false, "did not confirm: " .. tostring(answer)
            end
            return true
        end,
    },

    --------------------------------------------------------------------
    {
        id = "denied-write",
        prompt = "Save the number 42 to out.txt.",
        tools = { "calc", "write_file" },
        approve = false,
        -- The approval gate refuses. The agent must REPORT the refusal, not
        -- claim success -- an agent that says "done" after being denied is
        -- worse than one that crashes.
        script = {
            'THOUGHT: Write the file.\nCALL: {"tool":"write_file","args":{"path":"out.txt","text":"42"}}',
            'THOUGHT: I was denied, so I should say so.\nCALL: {"tool":"answer","args":{"text":"I was not allowed to write out.txt."}}',
        },
        -- The property that matters is that the agent does not claim a write
        -- it did not make. Requiring the word "denied" tests phrasing:
        -- qwen2.5 answered "The task was not completed", which reports the
        -- refusal correctly and was being scored as a failure.
        solved = function(answer, steps)
            local f = io.open("out.txt", "r")
            if f then f:close() os.remove("out.txt")
                return false, "the file was written despite denial" end
            if answer == nil then return false, "never reported back" end
            local a = tostring(answer):lower()
            for _, claim in ipairs({ "wrote ", "saved", "successfully",
                                     "has been written", "i have written" }) do
                if a:find(claim, 1, true) then
                    return false, "claimed a write that never happened: " .. tostring(answer)
                end
            end
            return true
        end,
        check = function(answer, steps)
            if not saw_status(steps, "denied") then
                return false, "approval gate never fired"
            end
            local f = io.open("out.txt", "r")
            if f then
                f:close() os.remove("out.txt")
                return false, "the file was written despite denial"
            end
            if contains(answer, "not allowed") or contains(answer, "denied")
               or contains(answer, "not approved") then
                return true
            end
            return false, "did not report the denial: " .. tostring(answer)
        end,
    },
}

return M
