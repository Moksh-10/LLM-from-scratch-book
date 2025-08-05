import torch
from torch import nn
from torch.linalg import diagonal

class mha(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "check d_out"

        self.dout = d_out
        self.nh = num_heads
        self.hd = d_out // num_heads

        self.wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.wk = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.wv = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.wo = nn.Linear(d_out, d_out)
        self.dr = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_len, context_len), diagonal=1))

    def forward(self, x):
        bs, sl, dim = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        k = k.view(bs, sl, self.nh, self.hd).transpose(1, 2)
        q = q.view(bs, sl, self.nh, self.hd).transpose(1, 2)
        v = v.view(bs, sl, self.nh, self.hd).transpose(1, 2)

        attn = q @ k.transpose(2, 3)
        mask_bool = self.mask.bool()[:sl, :sl]
        attn.masked_fill_(mask_bool, -torch.inf)
        attn = torch.softmax(attn / k.shape[-1] ** 0.5, dim=-1)
        attn = self.dr(attn)
        attn = (attn @ v).transpose(1, 2)
        attn = attn.reshape(bs, sl, self.dout)
        attn = self.wo(attn)
        return attn

