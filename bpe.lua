-- bpe.lua - Load lua-llama's BPE tokenizer without loading a model.
--
-- Exact token counts need the real tokenizer, and the real tokenizer is a
-- 433KB file. But repo 1's Tokenizer.new(path, vocab_size) takes the vocab
-- size as an argument, and the only place the rest of this repo had one was
-- model.config.vocab_size -- so counting tokens meant loading a 60MB
-- checkpoint and waiting ~17s for it.
--
-- That coupling is incidental: tokenizer.bin describes its own length, you
-- just have to walk it. This scans the file to derive the entry count, then
-- hands that to repo 1's loader. The replay and http backends get exact
-- counts with no model at all.
--
-- Format (llama2.c):
--   int32  max_token_length
--   repeated: float32 score, int32 len, len bytes

local bpe = {}

-- Walk the file and count entries. ~433KB and a few thousand reads, so it
-- costs milliseconds -- against 17s and 60MB for the checkpoint it replaces.
function bpe.count_entries(path)
    local fh, err = io.open(path, "rb")
    if not fh then return nil, tostring(err) end

    local header = fh:read(4)
    if not header or #header ~= 4 then
        fh:close()
        return nil, "truncated header"
    end

    local n = 0
    while true do
        local score = fh:read(4)
        if not score or #score < 4 then break end
        local len_bytes = fh:read(4)
        if not len_bytes or #len_bytes < 4 then
            fh:close()
            return nil, string.format("truncated length field at entry %d", n)
        end
        local len = string.unpack("<i4", len_bytes)
        if len < 0 or len > 1024 then
            fh:close()
            return nil, string.format("implausible token length %d at entry %d", len, n)
        end
        local tok = fh:read(len)
        if not tok or #tok ~= len then
            fh:close()
            return nil, string.format("truncated token body at entry %d", n)
        end
        n = n + 1
    end
    fh:close()

    if n == 0 then return nil, "no tokens found" end
    return n
end

-- Returns tokenizer, vocab_size  or  nil, err.
-- llama_path is where repo 1 lives, needed because its tokenizer module
-- requires('utils') by bare name.
function bpe.load(path, llama_path)
    local n, err = bpe.count_entries(path)
    if not n then return nil, err end

    local root = llama_path or "../lua_llama"
    package.path = root .. "/?.lua;" .. package.path

    local ok, Tokenizer = pcall(require, "tokenizer")
    if not ok then return nil, "cannot load lua-llama's tokenizer.lua: " .. tostring(Tokenizer) end

    -- Repo 1's loader prints as it goes; useful in its CLI, noise here.
    local real_print = print
    _G.print = function() end
    local loaded, tok = pcall(Tokenizer.new, path, n)
    _G.print = real_print

    if not loaded then return nil, tostring(tok) end
    return tok, n
end

-- The stand-in when no tokenizer file is available: roughly four characters
-- per token. Good enough to exercise the eviction policies, and never
-- reported as exact.
function bpe.approximate()
    return { encode = function(_, s)
        local out = {}
        for k = 1, math.max(1, math.ceil(#s / 4)) do out[k] = k end
        return out
    end }
end

return bpe
