import math
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# Reproducibility
random.seed(42)
torch.manual_seed(42)

# -------------------------
# 1) Tiny toy dataset
# -------------------------
pairs = [
    ("sub add { my ($a,$b)=@_; return $a+$b; }",
     "def add(a, b): return a + b"),

    ("sub subtract { my ($a,$b)=@_; return $a-$b; }",
     "def subtract(a, b): return a - b"),

    ("sub multiply { my ($a,$b)=@_; return $a*$b; }",
     "def multiply(a, b): return a * b"),

    ("sub divide { my ($a,$b)=@_; return $a/$b; }",
     "def divide(a, b): return a / b"),

    ("sub pow { my ($a,$b)=@_; return $a**$b; }",
     "def pow(a, b): return a ** b"),
]

# Shuffle and split (3 train, 1 val, hold out 1 for test; plus an unseen square)
random.shuffle(pairs)
train_pairs = pairs[:3]
val_pairs   = pairs[3:4]
test_pairs  = pairs[4:5] + [
    ("sub square { my ($a)=@_; return $a*$a; }",
     "def square(a): return a * a")
]

# -------------------------
# 2) Tokenizer & Vocab
# -------------------------
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD, SOS, EOS, UNK = range(4)

def tokenize(s: str) -> List[str]:
    # lightweight code-aware-ish tokenization
    punct = "(){}[];,:=+-*/%<>!^&|"
    for ch in punct:
        s = s.replace(ch, f" {ch} ")
    return s.split()

src2id = {tok: i for i, tok in enumerate(SPECIALS)}
tgt2id = {tok: i for i, tok in enumerate(SPECIALS)}

def add_to_vocab(pairs: List[Tuple[str, str]]):
    for src, tgt in pairs:
        for t in tokenize(src):
            if t not in src2id:
                src2id[t] = len(src2id)
        for t in tokenize(tgt):
            if t not in tgt2id:
                tgt2id[t] = len(tgt2id)

add_to_vocab(train_pairs + val_pairs + test_pairs)

id2src = {i: s for s, i in src2id.items()}
id2tgt = {i: s for s, i in tgt2id.items()}

def encode(tokens: List[str], vocab: dict, add_sos_eos=True) -> List[int]:
    ids = [vocab.get(t, UNK) for t in tokens]
    if add_sos_eos:
        return [SOS] + ids + [EOS]
    return ids

def pad_batch(seqs: List[List[int]], pad_value=PAD):
    max_len = max(len(s) for s in seqs)
    return torch.tensor([s + [pad_value]*(max_len-len(s)) for s in seqs], dtype=torch.long)

# -------------------------
# 3) Batching helpers
# -------------------------
def make_batch(pairs):
    src_batch, tgt_batch = [], []
    for src, tgt in pairs:
        src_batch.append(encode(tokenize(src), src2id, add_sos_eos=True))
        tgt_batch.append(encode(tokenize(tgt), tgt2id, add_sos_eos=True))
    return pad_batch(src_batch), pad_batch(tgt_batch)

train_src, train_tgt = make_batch(train_pairs)
val_src, val_tgt     = make_batch(val_pairs)

# -------------------------
# 4) Model: Mini Transformer
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1), :]
        return x

class MiniTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, nhead=4, num_layers=2, dim_ff=256, dropout=0.1):
        super().__init__()
        self.src_emb = nn.Embedding(len(src_vocab), d_model, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(len(tgt_vocab), d_model, padding_idx=PAD)
        self.pos_enc = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.out = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src_ids, tgt_ids):
        # src_ids/tgt_ids: [B, T]
        src_key_padding_mask = (src_ids == PAD)  # [B, T]
        tgt_key_padding_mask = (tgt_ids == PAD)

        # subsequent mask for decoder (no peek ahead)
        T_tgt = tgt_ids.size(1)
        tgt_mask = torch.triu(torch.ones(T_tgt, T_tgt, device=tgt_ids.device) * float("-inf"), diagonal=1)

        src = self.pos_enc(self.src_emb(src_ids))
        tgt = self.pos_enc(self.tgt_emb(tgt_ids))

        out = self.transformer(
            src, tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        return self.out(out)  # [B, T, V]

    @torch.no_grad()
    def greedy_decode(self, src_ids, max_len=64):
        self.eval()
        src_key_padding_mask = (src_ids == PAD)
        memory = self.transformer.encoder(
            self.pos_enc(self.src_emb(src_ids)),
            src_key_padding_mask=src_key_padding_mask
        )

        B = src_ids.size(0)
        ys = torch.full((B, 1), SOS, dtype=torch.long, device=src_ids.device)

        for _ in range(max_len):
            tgt_mask = torch.triu(torch.ones(ys.size(1), ys.size(1), device=ys.device) * float("-inf"), diagonal=1)
            tgt = self.pos_enc(self.tgt_emb(ys))
            out = self.transformer.decoder(
                tgt, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=(ys == PAD),
                memory_key_padding_mask=src_key_padding_mask
            )
            logits = self.out(out[:, -1, :])  # [B, V]
            next_token = logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == EOS).all():
                break
        return ys  # includes SOS ... EOS

# -------------------------
# 5) Train
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniTransformer(src2id, tgt2id, d_model=128, nhead=4, num_layers=2, dim_ff=256, dropout=0.1).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PAD)
optimizer = optim.AdamW(model.parameters(), lr=3e-3)

def run_epoch(src, tgt, train=True):
    if train:
        model.train()
    else:
        model.eval()

    # Teacher forcing: shift targets
    tgt_in  = tgt[:, :-1].to(device)
    tgt_out = tgt[:, 1:].to(device)
    src = src.to(device)

    logits = model(src, tgt_in)  # [B, T, V]
    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

    if train:
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return loss.item()

EPOCHS = 60  # small, runs fast on CPU
best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_src, train_tgt, train=True)
    val_loss   = run_epoch(val_src,   val_tgt,   train=False)

    if val_loss < best_val:
        best_val = val_loss
        best = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

# Load best weights (by val loss)
model.load_state_dict(best)

# -------------------------
# 6) Inference helpers
# -------------------------
def ids_to_text(ids: List[int], id2tok: dict) -> str:
    toks = []
    for i in ids:
        if i in (PAD, SOS, EOS):
            continue
        toks.append(id2tok.get(i, "<unk>"))
    return " ".join(toks)

def translate(perl_code: str) -> str:
    src_ids = pad_batch([encode(tokenize(perl_code), src2id, add_sos_eos=True)]).to(device)
    out_ids = model.greedy_decode(src_ids, max_len=64)[0].tolist()
    return ids_to_text(out_ids, id2tgt)

# -------------------------
# 7) Demo on test set
# -------------------------
for perl, gold in test_pairs:
    pred = translate(perl)
    print("\nPerl:", perl)
    print("Gold:", gold)
    print("Pred:", pred)