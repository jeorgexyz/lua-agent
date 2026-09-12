-- tmpfile.lua - A temporary path that can actually be opened.
--
-- os.tmpname() is not usable on Windows. It returns a bare name rooted at
-- the current drive -- "\s5tg." -- and opening it fails with permission
-- denied unless the process happens to be able to write to C:\. The backends
-- used it to stage curl's request body and header files, so every real
-- request died at `req:write` on a nil handle.
--
-- What makes this worth a module rather than a one-liner: the fixture tests
-- for backend/http.lua and backend/ollama.lua inject a transport, which is
-- the right way to test request building -- and it means they never execute
-- the curl path at all. No amount of that kind of testing would have found
-- this. It surfaced the first time a backend was pointed at a real endpoint.

local tmpfile = {}

local counter = 0

local function tempdir()
    -- TMPDIR is POSIX, TEMP/TMP are Windows. Fall back to the working
    -- directory, which is always writable if anything is.
    return os.getenv("TMPDIR") or os.getenv("TEMP") or os.getenv("TMP") or "."
end

-- Returns a path in a writable temp directory. Not opened, not created --
-- the caller writes it and is responsible for removing it.
function tmpfile.name(suffix)
    counter = counter + 1
    local dir = tempdir():gsub("[\\/]+$", "")
    local sep = package.config:sub(1, 1)
    -- os.time alone collides when two files are made in the same second,
    -- which is exactly what a request/header pair does.
    return string.format("%s%slua2_%d_%d_%d%s",
        dir, sep, os.time(), math.random(1e6), counter, suffix or "")
end

-- Write a string to a fresh temp file. Returns the path, or nil + error --
-- the callers need to report "could not stage the request" distinctly from
-- "the request failed", and silently proceeding on a nil handle is what
-- produced an index-a-nil-value crash instead of a message.
function tmpfile.write(contents, suffix)
    local path = tmpfile.name(suffix)
    local fh, err = io.open(path, "w")
    if not fh then
        return nil, string.format("cannot write a temp file at %s (%s)",
            path, tostring(err))
    end
    fh:write(contents)
    fh:close()
    return path
end

-- Read and delete in one step; used for curl's captured output.
function tmpfile.slurp(path)
    local fh = io.open(path, "r")
    if not fh then return nil end
    local s = fh:read("a")
    fh:close()
    return s
end

function tmpfile.remove(...)
    for _, p in ipairs({ ... }) do
        if p then os.remove(p) end
    end
end

return tmpfile
