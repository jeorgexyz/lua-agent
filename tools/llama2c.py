#!/usr/bin/env python3
"""Read a llama2.c checkpoint into PyTorch, and run it.

    python tools/llama2c.py ../lua_llama/stories15M.bin ../lua_llama/tokenizer.bin

This is the PyTorch counterpart to repo 1's model.lua and generate.lua. It
exists so a projection layer can be TRAINED against the same frozen model
the Lua side runs -- training needs autograd, which pure Lua does not have.

Correctness is not asserted, it is checked: at temperature 0 this must emit
the exact continuation repo 1 documents for "Once upon a time". Same
weights, same arithmetic, same output, or the loader is wrong. Run the file
to see it.

Format (7 int32 header, then float32 tensors, all layers contiguous):
    dim hidden_dim n_layers n_heads n_kv_heads vocab_size seq_len
    token_embedding, rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3,
    rms_final, freq_real, freq_imag, [wcls if vocab_size was negative]

A NEGATIVE vocab_size in the header means the classifier is NOT shared with
the embedding table -- the sign is the flag, which is easy to miss.
"""

import struct
import sys

import torch
import torch.nn.functional as F


class Config:
    def __init__(self, dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                 vocab_size, seq_len, shared_classifier):
        self.dim, self.hidden_dim = dim, hidden_dim
        self.n_layers, self.n_heads, self.n_kv_heads = n_layers, n_heads, n_kv_heads
        self.vocab_size, self.seq_len = vocab_size, seq_len
        self.shared_classifier = shared_classifier
        self.head_size = dim // n_heads
        self.kv_dim = n_kv_heads * self.head_size

    def __repr__(self):
        return (f"dim={self.dim} hidden={self.hidden_dim} layers={self.n_layers} "
                f"heads={self.n_heads} kv_heads={self.n_kv_heads} "
                f"vocab={self.vocab_size} seq_len={self.seq_len} "
                f"shared_cls={self.shared_classifier}")


def load(path):
    with open(path, "rb") as fh:
        blob = fh.read()

    hdr = struct.unpack("<7i", blob[:28])
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = hdr
    shared = vocab_size > 0
    vocab_size = abs(vocab_size)
    c = Config(dim, hidden_dim, n_layers, n_heads, n_kv_heads,
               vocab_size, seq_len, shared)

    pos = 28
    def take(*shape):
        nonlocal pos
        n = 1
        for s in shape:
            n *= s
        t = torch.frombuffer(blob, dtype=torch.float32, count=n,
                             offset=pos).clone().view(*shape)
        pos += n * 4
        return t

    w = {}
    w["tok_emb"] = take(vocab_size, dim)
    w["rms_att"] = take(n_layers, dim)
    w["wq"] = take(n_layers, dim, dim)
    w["wk"] = take(n_layers, c.kv_dim, dim)
    w["wv"] = take(n_layers, c.kv_dim, dim)
    w["wo"] = take(n_layers, dim, dim)
    w["rms_ffn"] = take(n_layers, dim)
    w["w1"] = take(n_layers, hidden_dim, dim)
    w["w2"] = take(n_layers, dim, hidden_dim)
    w["w3"] = take(n_layers, hidden_dim, dim)
    w["rms_final"] = take(dim)
    take(seq_len, c.head_size // 2)   # freq_cis_real, recomputed below
    take(seq_len, c.head_size // 2)   # freq_cis_imag

    if shared:
        w["wcls"] = w["tok_emb"]
    else:
        w["wcls"] = take(vocab_size, dim)

    return c, w


def rmsnorm(x, weight, eps=1e-5):
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def rope(x, pos, head_size):
    """Rotate pairs in place, llama2.c convention (adjacent pairs)."""
    out = x.clone().view(-1, head_size)
    idx = torch.arange(0, head_size, 2, dtype=torch.float32)
    freq = 1.0 / (10000.0 ** (idx / head_size))
    ang = pos * freq
    cos, sin = torch.cos(ang), torch.sin(ang)
    even, odd = out[:, 0::2].clone(), out[:, 1::2].clone()
    out[:, 0::2] = even * cos - odd * sin
    out[:, 1::2] = even * sin + odd * cos
    return out.view(-1)


@torch.no_grad()
def forward(c, w, token, pos, kcache, vcache, embed=None):
    """One token. `embed` overrides the token lookup -- that hook is the
    whole point of this file: a projected image vector is fed in exactly
    where a token embedding would be."""
    x = w["tok_emb"][token].clone() if embed is None else embed.clone()

    for l in range(c.n_layers):
        xb = rmsnorm(x, w["rms_att"][l])
        q = rope(w["wq"][l] @ xb, pos, c.head_size)
        k = rope(w["wk"][l] @ xb, pos, c.head_size)
        v = w["wv"][l] @ xb
        kcache[l, pos], vcache[l, pos] = k, v

        heads = []
        kv_mul = c.n_heads // c.n_kv_heads
        for h in range(c.n_heads):
            kvh = h // kv_mul
            qh = q[h * c.head_size:(h + 1) * c.head_size]
            ks = kcache[l, :pos + 1, kvh * c.head_size:(kvh + 1) * c.head_size]
            vs = vcache[l, :pos + 1, kvh * c.head_size:(kvh + 1) * c.head_size]
            att = torch.softmax((ks @ qh) / (c.head_size ** 0.5), dim=0)
            heads.append(att @ vs)
        x = x + w["wo"][l] @ torch.cat(heads)

        xb = rmsnorm(x, w["rms_ffn"][l])
        h1, h3 = w["w1"][l] @ xb, w["w3"][l] @ xb
        x = x + w["w2"][l] @ (F.silu(h1) * h3)

    return w["wcls"] @ rmsnorm(x, w["rms_final"])


def forward_seq(c, w, embeds):
    """Batched, differentiable forward over a whole sequence.

    embeds: [B, T, dim]  ->  logits: [B, T, vocab]

    The incremental forward above is what the Lua side mirrors; this one is
    for TRAINING. Gradients flow back through to `embeds`, which is how a
    projection layer learns: its output is spliced in as one position's
    embedding and the loss on a later token pushes it around. The frozen
    weights carry no grad -- only whatever produced `embeds` does.

    Kept honest by a test in __main__: the last position here must equal the
    incremental path's logits for the same input.
    """
    B, T, _ = embeds.shape
    x = embeds

    # Causal mask, built once.
    mask = torch.full((T, T), float("-inf"), device=x.device)
    mask = torch.triu(mask, diagonal=1)

    pos = torch.arange(T, device=x.device, dtype=torch.float32)
    idx = torch.arange(0, c.head_size, 2, device=x.device, dtype=torch.float32)
    freq = 1.0 / (10000.0 ** (idx / c.head_size))
    ang = pos[:, None] * freq[None, :]            # [T, head_size/2]
    cos, sin = torch.cos(ang), torch.sin(ang)

    def apply_rope(t, n_h):                        # t: [B, T, n_h, head_size]
        even, odd = t[..., 0::2], t[..., 1::2]
        c_ = cos[None, :, None, :]
        s_ = sin[None, :, None, :]
        out = torch.empty_like(t)
        out[..., 0::2] = even * c_ - odd * s_
        out[..., 1::2] = even * s_ + odd * c_
        return out

    for l in range(c.n_layers):
        xb = rmsnorm(x, w["rms_att"][l])
        q = (xb @ w["wq"][l].T).view(B, T, c.n_heads, c.head_size)
        k = (xb @ w["wk"][l].T).view(B, T, c.n_kv_heads, c.head_size)
        v = (xb @ w["wv"][l].T).view(B, T, c.n_kv_heads, c.head_size)
        q, k = apply_rope(q, c.n_heads), apply_rope(k, c.n_kv_heads)

        # Grouped-query attention: repeat each kv head for its query group.
        kv_mul = c.n_heads // c.n_kv_heads
        if kv_mul > 1:
            k = k.repeat_interleave(kv_mul, dim=2)
            v = v.repeat_interleave(kv_mul, dim=2)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))    # [B, nh, T, hs]
        att = (q @ k.transpose(-2, -1)) / (c.head_size ** 0.5) + mask
        out = (torch.softmax(att, dim=-1) @ v)
        out = out.transpose(1, 2).reshape(B, T, c.dim)
        x = x + out @ w["wo"][l].T

        xb = rmsnorm(x, w["rms_ffn"][l])
        x = x + (F.silu(xb @ w["w1"][l].T) * (xb @ w["w3"][l].T)) @ w["w2"][l].T

    return rmsnorm(x, w["rms_final"]) @ w["wcls"].T


def load_tokenizer(path, vocab_size):
    """Minimal reader for llama2.c tokenizer.bin -- enough to decode, and to
    encode with the same greedy merge repo 1 uses."""
    with open(path, "rb") as fh:
        blob = fh.read()
    pos = 4                       # max_token_length
    vocab, scores = [], []
    for _ in range(vocab_size):
        score = struct.unpack("<f", blob[pos:pos + 4])[0]; pos += 4
        ln = struct.unpack("<i", blob[pos:pos + 4])[0]; pos += 4
        vocab.append(blob[pos:pos + ln].decode("utf-8", "replace")); pos += ln
        scores.append(score)
    return vocab, scores


def decode(vocab, tid):
    piece = vocab[tid]
    if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
        return chr(int(piece[3:5], 16))
    return piece.replace("▁", " ")


def encode(vocab, scores, text):
    lookup = {}
    for i, v in enumerate(vocab):
        t = chr(int(v[3:5], 16)) if (len(v) == 6 and v.startswith("<0x")) else v
        lookup.setdefault(t, i)
    tokens = [1]                              # BOS
    for ch in (" " + text):
        tokens.append(lookup.get(ch, ord(ch) + 3))
    while True:
        best, best_id, best_at = -1e10, None, None
        for i in range(len(tokens) - 1):
            a, b = decode(vocab, tokens[i]), decode(vocab, tokens[i + 1])
            merged = lookup.get(a.replace(" ", "▁") + b.replace(" ", "▁"))
            if merged is None:
                merged = lookup.get(a + b)
            if merged is not None and scores[merged] > best:
                best, best_id, best_at = scores[merged], merged, i
        if best_at is None:
            break
        tokens[best_at] = best_id
        del tokens[best_at + 1]
    return tokens


@torch.no_grad()
def generate(c, w, vocab, scores, prompt, max_new=90):
    kcache = torch.zeros(c.n_layers, c.seq_len, c.kv_dim)
    vcache = torch.zeros(c.n_layers, c.seq_len, c.kv_dim)
    ids = encode(vocab, scores, prompt)

    pos, out = 0, []
    for t in ids[:-1]:
        forward(c, w, t, pos, kcache, vcache); pos += 1
    nxt = ids[-1]
    for _ in range(max_new):
        logits = forward(c, w, nxt, pos, kcache, vcache)
        nxt = int(torch.argmax(logits))          # temperature 0
        if nxt in (1, 2):
            break
        out.append(decode(vocab, nxt))
        pos += 1
        if pos >= c.seq_len:
            break
    return prompt + "".join(out)


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "../lua_llama/stories15M.bin"
    tok = sys.argv[2] if len(sys.argv) > 2 else "../lua_llama/tokenizer.bin"

    c, w = load(ckpt)
    print(c)
    vocab, scores = load_tokenizer(tok, c.vocab_size)
    print(f"tokenizer: {len(vocab)} entries\n")

    text = generate(c, w, vocab, scores, "Once upon a time", max_new=60)
    print(text)

    # Repo 1's README documents this continuation at temperature 0. Same
    # weights and same arithmetic must give the same text.
    expected = "there was a little girl named Lily"
    print("\nmatches repo 1's documented output:", expected in text)

    # The batched path is what training runs through, so it has to agree
    # with the incremental one it will never otherwise be compared against.
    ids = encode(vocab, scores, "Once upon a time")
    emb = w["tok_emb"][torch.tensor(ids)].unsqueeze(0)
    seq_logits = forward_seq(c, w, emb)[0, -1]

    kc = torch.zeros(c.n_layers, c.seq_len, c.kv_dim)
    vc = torch.zeros(c.n_layers, c.seq_len, c.kv_dim)
    for p, t in enumerate(ids[:-1]):
        forward(c, w, t, p, kc, vc)
    inc_logits = forward(c, w, ids[-1], len(ids) - 1, kc, vc)

    delta = (seq_logits - inc_logits).abs().max().item()
    print(f"batched vs incremental: max |diff| = {delta:.2e}, "
          f"same argmax = {int(seq_logits.argmax()) == int(inc_logits.argmax())}")

    # And gradients must reach the input embeddings, or nothing can train.
    probe = emb.clone().requires_grad_(True)
    forward_seq(c, w, probe)[0, -1].sum().backward()
    g = probe.grad.abs().sum().item()
    print(f"gradient reaches the input embedding: {g > 0} (|grad| sum {g:.3f})")
