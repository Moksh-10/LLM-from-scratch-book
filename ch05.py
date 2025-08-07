from ch04 import gen_text
import json
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
from torch import nn
from tqdm import tqdm


def gen(model, idx, max_new_tok, context_size, temp=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tok):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        if temp > 0.0:
            logits = logits / temp
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def text_to_token_ids(text, tokenizer):
    enc = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    enc_ten = torch.tensor(enc).unsqueeze(0)
    return enc_ten


def token_ids_tok_text(token_ids, tokenizer):
    dec = token_ids.squeeze(0)
    return tokenizer.decode(dec.tolist())


def calc_loss_batch(inp_batch, tar_batch, model, device):
    inp_batch, tar_batch = inp_batch.to(device), tar_batch.to(device)
    logits = model(inp_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), tar_batch.flatten())
    return loss



