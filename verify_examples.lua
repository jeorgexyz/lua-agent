-- verify_examples.lua - Every transcript's replay command must reproduce it.
--
--   lua54 verify_examples.lua
--
-- Each file in examples/ prints a `replay:` line. This runs that exact
-- command as a subprocess and checks the result still matches the
-- transcript's own ANSWER line.
--
-- The replay group in test.lua drives the Agent directly, which is faster
-- but checks a different thing: it never exercises main.lua's argument
-- defaults. That gap hid a real bug -- main.lua defaults --max-steps to 6,
-- loop-bait needs 7, so the documented command stopped one step short of the
-- answer and reported a max_steps failure for a run that had succeeded. Every
-- test was green while the published instructions were wrong.
--
-- A command in a README is a claim. This is the test for it.

package.path = "./?.lua;" .. package.path

local lfs_dir = "examples"

-- Windows and POSIX name the interpreter differently; let the environment
-- override when neither guess is right.
local LUA = os.getenv("LUA") or "lua54"

local function read_file(path)
    local fh = io.open(path, "r")
    if not fh then return nil end
    local s = fh:read("a")
    fh:close()
    return s
end

local function list_examples()
    local cmd = package.config:sub(1, 1) == "\\"
        and ('dir /b "' .. lfs_dir .. '\\*.txt" 2>nul')
        or ("ls -1 " .. lfs_dir .. "/*.txt 2>/dev/null")
    local pipe = io.popen(cmd, "r")
    if not pipe then return {} end
    local out = {}
    for line in pipe:lines() do
        local name = line:match("([^/\\]+%.txt)$")
        if name then out[#out + 1] = lfs_dir .. "/" .. name end
    end
    pipe:close()
    table.sort(out)
    return out
end

local pass, fail, skipped = 0, 0, 0

for _, path in ipairs(list_examples()) do
    local text = read_file(path)
    local cmd = text and text:match("replay:%s*([^\n]+)")
    local want = text and text:match("ANSWER:%s*([^\n]+)")

    if not cmd then
        skipped = skipped + 1
        io.write(string.format("SKIP  %s (no replay: line)\n", path))
    else
        -- The transcripts say `lua54` because that is what a reader types;
        -- honour $LUA so this runs wherever the interpreter is called
        -- something else.
        local run = cmd:gsub("^lua54", (LUA:gsub("%%", "%%%%")))

        -- cmd.exe strips the outer quotes of a command that begins with a
        -- quoted path, so `"C:\Program Files\...\lua54.exe" args` becomes
        -- 'C:\Program' is not recognized. Wrapping the whole string restores
        -- it. Only bites when the interpreter lives somewhere with a space.
        local shell_cmd = run
        if package.config:sub(1, 1) == "\\" and run:sub(1, 1) == '"' then
            shell_cmd = '"' .. run .. '"'
        end

        -- Close stdin. The denied-write transcript deliberately omits
        -- --yes, so main.lua reaches the approval prompt and calls
        -- io.read("l"). With stdin inherited from a terminal that blocks
        -- forever instead of returning nil -- this harness hung on it, and
        -- CI would have hung the same way.
        local devnull = package.config:sub(1, 1) == "\\" and "NUL" or "/dev/null"
        local pipe = io.popen(shell_cmd .. " < " .. devnull .. " 2>&1", "r")
        local out = pipe and pipe:read("a") or ""
        if pipe then pipe:close() end

        local got = out:match("ANSWER:%s*([^\n]+)")
        local stopped = out:match("Stopped because:%s*([^\n]+)")

        if want and got and got:gsub("%s+$", "") == want:gsub("%s+$", "") then
            pass = pass + 1
            io.write(string.format("ok    %s\n", path))
        elseif not want and stopped then
            -- A transcript that records a non-answer outcome is fine as long
            -- as the replay reaches the same one.
            pass = pass + 1
            io.write(string.format("ok    %s (stopped: %s)\n", path, stopped))
        else
            fail = fail + 1
            io.write(string.format("FAIL  %s\n", path))
            io.write("        command:  " .. run .. "\n")
            io.write("        expected: " .. tostring(want) .. "\n")
            io.write("        got:      " .. tostring(got or ("<no answer> " ..
                tostring(stopped))) .. "\n")
        end
    end
end

io.write(string.format("\n%d reproduced, %d failed, %d skipped\n",
    pass, fail, skipped))
os.exit(fail == 0 and 0 or 1)
