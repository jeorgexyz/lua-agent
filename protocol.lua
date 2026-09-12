-- protocol.lua - Wire format between the loop and the model.
--
-- The whole "tool calling" abstraction is this file: a way to render tools as
-- text, and a way to read a tool call back out of text. No provider magic.
--
-- Two decisions here shape everything downstream:
--
-- 1. ANSWERING IS A TOOL CALL. There is no separate ANSWER verb; finishing is
--    {"tool":"answer","args":{"text":"..."}}. One grammar covers every turn,
--    the loop has one exit, and a constrained model can always terminate --
--    which it could not if answering lived outside the grammar.
--
-- 2. EVERY ARGUMENT IS A STRING on the wire. Tools declare number types and
--    the registry coerces on the way in, but the model only ever writes
--    quoted values. This removes an entire state from grammar.lua (numbers
--    are the fiddly case: they have no closing delimiter, so the machine
--    cannot tell "done" from "more digits" without lookahead).

local trace = require('trace')

local protocol = {}

-- Surface syntax. Line-delimited and JSON-shaped, because both properties
-- make grammar.lua's job tractable.
--
--   THOUGHT: <free text, one line>
--   CALL: {"tool":"<name>","args":{...}}
protocol.THOUGHT_PREFIX = "THOUGHT: "
protocol.CALL_PREFIX    = "CALL: "
protocol.OBS_PREFIX     = "RESULT: "
protocol.ANSWER_TOOL    = "answer"

-- Where an unconstrained backend should stop generating.
protocol.STOP = { "\nRESULT:", "\nTHOUGHT:" }

-- The pseudo-tool that terminates the loop. Registered automatically by
-- tools.registry() so it appears in the catalogue and in the grammar enum.
protocol.answer_tool = {
    name = protocol.ANSWER_TOOL,
    description = "Give the final answer and stop. Use when the task is done.",
    args = { { name = "text", type = "string", required = true } },
    run = function(args) return args.text end,
}

-- Render one tool as the model sees it. This is prompt real estate charged
-- against the context budget on every single step, so it stays terse.
function protocol.render_tool(tool)
    local parts = {}
    for _, a in ipairs(tool.args or {}) do
        parts[#parts + 1] = string.format('"%s":"<%s>"', a.name,
            a.required and a.type or (a.type .. ", optional"))
    end
    return string.format('  %s -- %s\n    {"tool":"%s","args":{%s}}',
        tool.name, tool.description, tool.name, table.concat(parts, ","))
end

-- A catalogue terse enough for a tiny context window.
--
-- This is not a micro-optimization; it is forced. stories15M has seq_len=256,
-- and the verbose catalogue below costs 194 tokens for two tools and 262 for
-- four -- so with the default tool set the system prompt alone overflows the
-- model's entire context before the task is even stated.
--
-- Discovering that is one of the more useful things about driving a 15M
-- model: context budgeting stops being a hypothetical you handle at 100k
-- tokens and becomes the first thing that breaks. The compact form drops the
-- format tutorial (the grammar enforces the format anyway, so spending 90
-- tokens explaining it to a constrained model buys nothing) and renders each
-- tool as one line.
function protocol.render_compact(tools, task)
    local lines = { "Tools:" }
    for _, t in ipairs(tools:list()) do
        local keys = {}
        for _, a in ipairs(t.args or {}) do
            if a.required then keys[#keys + 1] = '"' .. a.name .. '"' end
        end
        lines[#lines + 1] = string.format("%s{%s} %s",
            t.name, table.concat(keys, ","), t.description)
    end
    lines[#lines + 1] = "TASK: " .. task
    return table.concat(lines, "\n")
end

-- The system preamble: framing + tool catalogue + the task.
-- Kept separate from the transcript so context.lua can price it as a fixed
-- cost that is never evicted.
--
-- `compact` selects the terse catalogue above -- required for any model with
-- a small context window.
function protocol.render_system(tools, task, compact)
    if compact then return protocol.render_compact(tools, task) end
    local lines = {
        "You are an agent. Work in steps. At each step write one thought and",
        "then exactly one tool call, in this format:",
        "",
        protocol.THOUGHT_PREFIX .. "<why you are doing this>",
        protocol.CALL_PREFIX .. '{"tool":"<name>","args":{...}}',
        "",
        "The result appears as " .. protocol.OBS_PREFIX .. "<output>, then you go again.",
        "All argument values are strings. Available tools:",
        "",
    }
    for _, t in ipairs(tools:list()) do
        lines[#lines + 1] = protocol.render_tool(t)
    end
    lines[#lines + 1] = ""
    lines[#lines + 1] = "TASK: " .. task
    return table.concat(lines, "\n")
end

-- Render the running transcript: alternating model turns and observations.
function protocol.render_transcript(steps)
    local out = {}
    for _, s in ipairs(steps) do
        if s.thought and s.thought ~= "" then
            out[#out + 1] = protocol.THOUGHT_PREFIX .. s.thought
        end
        if s.call then
            out[#out + 1] = protocol.CALL_PREFIX .. trace.encode({
                tool = s.call.tool, args = s.call.args or {},
            })
        elseif s.raw then
            out[#out + 1] = s.raw
        end
        if s.observation then
            out[#out + 1] = protocol.OBS_PREFIX .. s.observation
        end
        if s.note then
            out[#out + 1] = s.note
        end
    end
    return table.concat(out, "\n")
end

-- Parse model output into a structured turn.
-- Returns one of:
--   { kind = "call",   tool = , args = , thought = }
--   { kind = "answer", text = ,          thought = }
--   nil, err_message
--
-- err_message is written to be read BY THE MODEL -- it goes back as the next
-- observation, and is the entire mechanism of parse recovery. So it says what
-- was wrong and what correct output looks like, not "parse error at byte 34".
function protocol.parse(text)
    if type(text) ~= "string" then
        return nil, "No output was produced."
    end

    local thought = text:match(protocol.THOUGHT_PREFIX .. "([^\n]*)")
    if thought then thought = thought:match("^%s*(.-)%s*$") end

    -- Take the LAST call in the output. A model that restates the format from
    -- the prompt before committing to a call is common; the last one is the
    -- one it meant.
    local payload
    for m in text:gmatch(protocol.CALL_PREFIX .. "([^\n]*)") do payload = m end
    if not payload then
        return nil, string.format(
            "No tool call found. Every step must contain a line starting with %q, "
            .. "for example: %s{\"tool\":\"answer\",\"args\":{\"text\":\"...\"}}",
            protocol.CALL_PREFIX, protocol.CALL_PREFIX)
    end

    payload = payload:match("^%s*(.-)%s*$")
    local obj, err = trace.decode(payload)
    if not obj then
        return nil, string.format(
            "The tool call was not valid JSON (%s). Write it on one line as "
            .. '{"tool":"<name>","args":{"<key>":"<value>"}}.', err)
    end
    if type(obj) ~= "table" then
        return nil, 'The tool call must be a JSON object starting with {"tool":.'
    end
    if type(obj.tool) ~= "string" then
        return nil, 'The tool call is missing a "tool" name, e.g. {"tool":"calc","args":{...}}.'
    end
    if obj.args ~= nil and type(obj.args) ~= "table" then
        return nil, '"args" must be a JSON object, e.g. "args":{"expr":"2+2"}.'
    end

    local args = obj.args or {}
    if obj.tool == protocol.ANSWER_TOOL then
        if type(args.text) ~= "string" then
            return nil, 'The answer call needs a text argument: '
                .. '{"tool":"answer","args":{"text":"<your answer>"}}.'
        end
        return { kind = "answer", text = args.text, thought = thought }
    end
    return { kind = "call", tool = obj.tool, args = args, thought = thought }
end

return protocol
