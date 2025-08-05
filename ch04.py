from ch03 import mha
import torch
from torch import nn


class gelu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class ln(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm + self.shift
    

class ff(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.l = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            gelu(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
        
    def forward(self, x):
        return self.l(x)
    

class trans_f(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = mha(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_len=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = ff(cfg)
        self.ln1 = ln(cfg["emb_dim"])
        self.ln2 = ln(cfg["emb_dim"])
        self.dr = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.dr(self.att(self.ln1(x)))
        x = x + self.dr(self.ff(self.ln2(x)))
        return x


class gm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_lenght"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.tr_l = nn.Sequential(*[trans_f(cfg) for _ in range(cfg["n_layers"])])
        self.fn = ln(cfg["emb_dim"])
        self.out = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        bs, sl = in_idx.shape
        tok_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(torch.arange(sl, device=in_idx.device))
        x = tok_emb + pos_emb
        x = self.drop_emb(x)
        x = self.tr_l(x)
        x = self.fn(x)
        logits = self.out(x)
        return logits


def gen_text(model, idx, max_new_tok, context_size):
    for _ in range(max_new_tok):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
