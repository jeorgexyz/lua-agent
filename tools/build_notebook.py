#!/usr/bin/env python3
"""Generate colab/train_projection.ipynb.

The notebook is generated rather than hand-written for the same reason the
demo GIFs are: so it cannot drift from the code it depends on, and so the
cell sources stay readable as Python instead of as escaped JSON.

    python tools/build_notebook.py
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "colab", "train_projection.ipynb")

MD_INTRO = """# Training a LLaVA-style projection for `lua-agent`

**What this trains:** one matrix. Everything else is frozen.

| Piece | Params | Trained here |
|---|---|---|
| CLIP ViT-B/32 vision encoder | 88M | no — frozen |
| `stories15M` language model | 15M | no — frozen |
| **projection, 512 -> 288** | **~147K** | **yes** |

That is the whole idea of a multimodal *connector*: the two towers already
work, and the only thing missing is a map from one embedding space into the
other. LLaVA trains exactly this (on far more data, projecting 576 patch
tokens instead of one pooled vector).

**The experiment.** Feed the projected image vector in where a token
embedding would go, prompt `a photo of a`, and train the projection so the
frozen LLM's next token is the CIFAR-10 class word. Then read accuracy out
of the generated text.

**Chance is 10%.** Anything meaningfully above it means the projection
learned to steer a model that has never seen an image. Anything at it means
a 15M model trained only on children's stories cannot be steered this way —
which is a real result too, and the same shape as finding that `stories42M`
could not pick a tool.

**Honest simplifications**, both worth knowing before reading the number:

- One pooled CLIP vector, not 576 patch tokens. The simplest possible
  connector.
- CIFAR-10 is 32x32 upscaled to 224. CLIP handles it, but it is not a
  natural-image benchmark.

Runtime: a few minutes on a free T4. Set **Runtime -> Change runtime type ->
T4 GPU**, though CPU also finishes.
"""

CELL_SETUP = '''# Setup: weights, tokenizer, and the loader this repo already verifies.
!pip -q install transformers torchvision

!wget -q -nc https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
!wget -q -nc https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
!wget -q -nc https://raw.githubusercontent.com/jeorgexyz/lua-agent/main/tools/llama2c.py

import torch, llama2c
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
'''

CELL_GONOGO = '''# GO / NO-GO. Load the frozen LLM and check it against a known-good output.
#
# repo 1 documents this exact continuation at temperature 0. If the loader
# is wrong, this is where you find out -- before building anything on it.

cfg, W = llama2c.load("stories15M.bin")
vocab, scores = llama2c.load_tokenizer("tokenizer.bin", cfg.vocab_size)
print(cfg)

text = llama2c.generate(cfg, W, vocab, scores, "Once upon a time", max_new=40)
print("\\n" + text)
assert "there was a little girl named Lily" in text, "loader mismatch -- stop here"
print("\\nloader verified against repo 1's documented output")
'''

CELL_CLASSES = '''# The target is the FIRST token of the real class name.
#
# Only six of the ten CIFAR names are single tokens here. The tempting fix
# is single-token synonyms, but the only ones available are "sheep" for deer
# and "fish" for frog -- which would train the projection on labels that are
# simply wrong and quietly make the accuracy number meaningless.
#
# Predicting the first token keeps the real names. It is well posed as long
# as those ten first tokens are distinct, which is asserted below rather
# than assumed.
#
# Read the number knowing that " de", " f" and " tr" are generic fragments:
# emitting " cat" is stronger evidence than emitting " de". The per-class
# breakdown in the eval cell is where that shows up.

CIFAR = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

CLASS_IDS = []
for w in CIFAR:
    ids = llama2c.encode(vocab, scores, w)[1:]        # drop BOS
    CLASS_IDS.append(ids[0])
    pieces = [llama2c.decode(vocab, t) for t in ids]
    print(f"  {w:<11} -> {pieces}  target={ids[0]}")

assert len(set(CLASS_IDS)) == 10, "two classes share a first token -- not well posed"
print("\\nall ten first tokens are distinct")
'''

CELL_EMBED = '''# Embed CIFAR-10 once with frozen CLIP. This is the slow part; everything
# after it trains on cached vectors.

import torch, numpy as np
from torchvision import datasets, transforms
from transformers import CLIPModel, CLIPImageProcessor

dev = "cuda" if torch.cuda.is_available() else "cpu"
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

mean, std = proc.image_mean, proc.image_std
tf = transforms.Compose([
    transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

def embed(train, limit):
    ds = datasets.CIFAR10("./data", train=train, download=True, transform=tf)
    ds = torch.utils.data.Subset(ds, range(min(limit, len(ds))))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=2)
    xs, ys = [], []
    with torch.no_grad():
        for img, lab in dl:
            f = clip.get_image_features(pixel_values=img.to(dev))
            xs.append(torch.nn.functional.normalize(f, dim=-1).cpu())
            ys.append(lab)
    return torch.cat(xs), torch.cat(ys)

Xtr, Ytr = embed(True, 20000)
Xte, Yte = embed(False, 2000)
print("train", tuple(Xtr.shape), "| test", tuple(Xte.shape))
'''

CELL_TRAIN = '''# Train the projection. One matrix, 512 -> 288.
#
# Sequence fed to the frozen LLM:   [BOS] [projected image] a photo of a ?
# Loss: cross-entropy on the single next token against the class word.
# Gradients reach only the projection -- llama2c.forward_seq is
# differentiable w.r.t. its input embeddings, which is the hook that makes
# a connector trainable at all.

import torch.nn as nn, torch.nn.functional as F

Wd = {k: v.to(dev) for k, v in W.items()}
for v in Wd.values():
    v.requires_grad_(False)

PROMPT = llama2c.encode(vocab, scores, "a photo of a")   # begins with BOS
prompt_ids = torch.tensor(PROMPT[1:], device=dev)        # drop BOS, re-added below
bos = Wd["tok_emb"][1]

proj = nn.Sequential(nn.Linear(512, cfg.dim)).to(dev)    # the only trainable thing
opt = torch.optim.AdamW(proj.parameters(), lr=1e-3, weight_decay=0.0)
print("trainable params:", sum(p.numel() for p in proj.parameters()))

def build(batch_x):
    B = batch_x.shape[0]
    img = proj(batch_x.to(dev)).unsqueeze(1)             # [B, 1, dim]
    pre = bos.expand(B, 1, cfg.dim)
    txt = Wd["tok_emb"][prompt_ids].unsqueeze(0).expand(B, -1, cfg.dim)
    return torch.cat([pre, img, txt], dim=1)             # [B, T, dim]

EPOCHS, BS = 3, 64
for ep in range(EPOCHS):
    perm = torch.randperm(len(Xtr))
    total = n = 0
    for i in range(0, len(perm), BS):
        idx = perm[i:i + BS]
        emb = build(Xtr[idx])
        logits = llama2c.forward_seq(cfg, Wd, emb)[:, -1]  # predict next
        target = torch.tensor([CLASS_IDS[y] for y in Ytr[idx]], device=dev)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(idx); n += len(idx)
    print(f"epoch {ep+1}  loss {total/n:.4f}")
'''

CELL_EVAL = '''# Read accuracy out of the model's own next token -- not out of the logits
# restricted to the ten class ids. Restricting would flatter the result: it
# measures "which of ten" when the claim is "the frozen LLM says the word".

import torch

@torch.no_grad()
def evaluate(X, Y):
    correct = free_correct = 0
    for i in range(0, len(X), 256):
        emb = build(X[i:i + 256])
        logits = llama2c.forward_seq(cfg, Wd, emb)[:, -1]
        target = torch.tensor([CLASS_IDS[y] for y in Y[i:i + 256]], device=dev)

        free_correct += (logits.argmax(-1) == target).sum().item()   # whole vocab
        among = logits[:, CLASS_IDS].argmax(-1).cpu()                # forced choice
        correct += (torch.tensor([CLASS_IDS[j] for j in among],
                                 device=dev) == target).sum().item()
    return free_correct / len(X), correct / len(X)

free, forced = evaluate(Xte, Yte)
print(f"free generation (whole 32000-token vocabulary): {free:6.1%}")
print(f"forced choice among the 10 class tokens:        {forced:6.1%}")
print(f"chance:                                          10.0%")
print()
print("The first number is the honest one: it is what the agent would")
print("actually see the model emit. The second says whether the signal is")
print("present at all. A large gap means the projection learned something")
print("the frozen model will not say unprompted.")

# Per class, because three targets are generic fragments (" de", " f",
# " tr") that are easier to hit than " cat" or " horse". If the accuracy is
# carried by those three, the headline number is flattering itself.
print("\\nper class (forced choice):")
with torch.no_grad():
    for ci, name in enumerate(CIFAR):
        sel = (Yte == ci).nonzero().flatten()
        if len(sel) == 0:
            continue
        hit = 0
        for i in range(0, len(sel), 256):
            idx = sel[i:i + 256]
            lg = llama2c.forward_seq(cfg, Wd, build(Xte[idx]))[:, -1]
            hit += (lg[:, CLASS_IDS].argmax(-1).cpu() == ci).sum().item()
        tok = repr(llama2c.decode(vocab, CLASS_IDS[ci]))
        print(f"  {name:<11} {tok:<10} {hit/len(sel):6.1%}  (n={len(sel)})")
'''

CELL_EXPORT = '''# Export for the Lua side: float32, row-major, same convention as
# llama2.c so the reader is the one repo 1 already has.
#
#   int32 in_dim, int32 out_dim, then out_dim*in_dim weights, then out_dim bias

import struct

lin = proj[0]
Wp = lin.weight.detach().cpu().numpy().astype("float32")   # [out, in]
Bp = lin.bias.detach().cpu().numpy().astype("float32")

with open("projection.bin", "wb") as f:
    f.write(struct.pack("<2i", Wp.shape[1], Wp.shape[0]))
    f.write(Wp.tobytes()); f.write(Bp.tobytes())

print("wrote projection.bin", Wp.shape, "+ bias", Bp.shape)
from google.colab import files; files.download("projection.bin")
'''

MD_NEXT = """## What to do with the result

**If accuracy is well above 10%:** the connector works, and the Lua side is
worth building — a ViT forward pass, this projection, and the existing
`generate` with one embedding spliced in. The number goes in the README the
same way the 100%/0% parse-rate table did.

**If it sits at chance:** that is the finding. A 15M model trained only on
children's stories cannot be steered by a projected image vector, and the
honest write-up is the negative result plus the two obvious next moves —
`stories110M`, or patch tokens instead of one pooled vector.

Either way the experiment is cheap and the claim is falsifiable, which is
the bar the rest of this project is held to.
"""


def cells():
    yield {"cell_type": "markdown", "metadata": {},
           "source": MD_INTRO.splitlines(keepends=True)}
    for src in (CELL_SETUP, CELL_GONOGO, CELL_CLASSES, CELL_EMBED,
                CELL_TRAIN, CELL_EVAL, CELL_EXPORT):
        yield {"cell_type": "code", "metadata": {}, "outputs": [],
               "execution_count": None, "source": src.splitlines(keepends=True)}
    yield {"cell_type": "markdown", "metadata": {},
           "source": MD_NEXT.splitlines(keepends=True)}


nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": list(cells()),
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", os.path.relpath(OUT, ROOT), "-", len(nb["cells"]), "cells")
