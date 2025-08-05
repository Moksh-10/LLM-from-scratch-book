from ch04 import gen_text
import json
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
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


def train(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    trian_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for inp_batch, tar_batch in train_loader:
            optimizer.zero_grad()
            loss = loss_batch(inp_batch, tar_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += inp_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = eval_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                trian_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch: {epoch + 1} (Step {global_step:06g}): "
                      f"Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")

        gen_and_print(model, tokenizer, device, start_context)

    return trian_losses, val_losses, track_tokens_seen


def eval_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


