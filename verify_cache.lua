-- verify_cache.lua - The KV cache reuse invariant.
--
--   lua54 verify_cache.lua [--checkpoint ../lua_llama/stories15M.bin]
--
-- backend/local.lua does two things that are only safe if stale cache
-- entries are truly unreachable:
--
--   1. `feed` rewinds `pos` on a prefix divergence instead of clearing the
--      cache past that point.
--   2. `reset` allocates the cache once and afterwards only zeroes `pos`,
--      rather than re-zeroing 8.4M numbers per task.
--
-- Both rest on the same claim: the cache is position-indexed, and every slot
-- below `pos` is written before it is read again. If that is wrong, output
-- silently degrades rather than crashing -- the worst failure mode there is.
--
-- So this checks it the way repo 1 checked its speculative draft path: at
-- temperature 0, a warm run and a cold run must produce byte-identical text.
-- A mismatch here is the bug that KV-cache reuse always has.

package.path = "./?.lua;" .. package.path

local tools    = require('tools')
local protocol = require('protocol')
local blocal   = require('backend.local')

local opts = {}
for i = 1, #arg, 2 do opts[(arg[i]:gsub("^%-%-", ""))] = arg[i + 1] end

local reg = tools.registry()
reg:add(tools.calc)
reg:add(tools.read_file)

local PROMPTS = {
    protocol.render_compact(reg, "What is 4871 * 209?"),
    protocol.render_compact(reg, "Read the notes file."),
    protocol.render_compact(reg, "Add 17 and 25."),
}

local function new_backend()
    return blocal.new({ tools = reg, constrain = true, temperature = 0.0,
                        checkpoint = opts.checkpoint, llama_path = opts.llama_path })
end

io.write("loading...\n") io.flush()
local warm = new_backend()

-- Warm: one backend, all prompts in sequence, reset between them. This
-- exercises both the allocate-once reset and the prefix rewind in feed.
local warm_out = {}
for i, p in ipairs(PROMPTS) do
    warm:reset()
    warm_out[i] = warm:complete(p, {})
    io.write(string.format("  warm %d done\n", i)) io.flush()
end

-- Interleaved: same backend, prompts in a different order, then the originals
-- again. If a stale entry were reachable, a differing history would surface it.
warm:reset() warm:complete(PROMPTS[3], {})
warm:reset() warm:complete(PROMPTS[2], {})
local interleaved = {}
for i, p in ipairs(PROMPTS) do
    warm:reset()
    interleaved[i] = warm:complete(p, {})
end

-- Cold: a brand-new backend per prompt. The ground truth.
local cold_out = {}
for i, p in ipairs(PROMPTS) do
    local fresh = new_backend()
    cold_out[i] = fresh:complete(p, {})
    io.write(string.format("  cold %d done\n", i)) io.flush()
end

local fail = 0
io.write("\n")
for i = 1, #PROMPTS do
    local w_ok = warm_out[i] == cold_out[i]
    local x_ok = interleaved[i] == cold_out[i]
    if not w_ok or not x_ok then fail = fail + 1 end
    io.write(string.format("prompt %d: warm==cold %s   interleaved==cold %s\n",
        i, w_ok and "yes" or "NO", x_ok and "yes" or "NO"))
    if not (w_ok and x_ok) then
        io.write("  cold:        " .. cold_out[i]:gsub("\n", " | ") .. "\n")
        io.write("  warm:        " .. warm_out[i]:gsub("\n", " | ") .. "\n")
        io.write("  interleaved: " .. interleaved[i]:gsub("\n", " | ") .. "\n")
    end
end

io.write("\n")
if fail == 0 then
    io.write("PASS: cache reuse is transparent at temperature 0\n")
else
    io.write(string.format("FAIL: %d of %d prompts diverged\n", fail, #PROMPTS))
end
os.exit(fail == 0 and 0 or 1)
