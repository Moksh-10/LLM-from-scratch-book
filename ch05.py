import requests

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


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
         num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, tar_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, tar_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def eval(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss


def train(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for inp_batch, tar_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(inp_batch, tar_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += inp_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = eval(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch: {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
        gen_and_print(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


def gen_and_print(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    enc = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = gen_text(model, enc, max_new_tok=50, context_size=context_size)
        dec_text = token_ids_tok_text(token_ids, tokenizer)
        print(dec_text.replace("\n", " "))
    model.train()


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return nn.Parameter(torch.tensor(right))


def plot_loss(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplot(figsize=(5, 3))

    ax1.plot(epochs_seen, train_losses, label="train loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="val loss")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.savefig("loss-plot.pdf")
    plt.show()


def load_wei_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def downaload_and_load_gpt2(model_size, models_dir):
    import tensorflow as tf

    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"model size not in {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(models_dir, filename)
        download_file(file_url, file_path, backup_url)

        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
        settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
        params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

        return settings, params


def download_file(url, destination, backup_url=None):
    def _attempt_download(download_url):
        res = requests.get(download_url, stream=True, timeout=60)
        res.raise_for_status()

        fs = int(res.headers.get("Content-Length", 0))

        if os.path.exists(destination):
            fs_local = os.path.getsize(destination)
            if fs and fs == fs_local:
                print(f"file already exists and is up to date: {destination}")
                return True

        bs = 1024
        desc = os.path.basename(download_url)
        with tqdm(total=fs, unit="iB", unit_scale=True, desc=desc) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in res.iter_content(chunk_size=bs):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))

        return True

    try:
        if _attempt_download(url):
            return
    except requests.exceptions.RequestException:
        if backup_url is not None:
            print(f"primary url ({url}) failes. trying with: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except requests.exceptions.RequestException:
                pass

        error_msg = (
            f"failed with both urls"
        )
        print(error_msg)
    except Exception as e:
        print(f"error: {e}")


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    import tensorflow as tf

    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        var_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        var_name_parts = name.split("/")[1:]

        target_dict = params
        if var_name_parts[0].startswith("h"):
            layer_no = int(var_name_parts[0][1:])
            target_dict = params["blocks"][layer_no]

        for key in var_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = var_name_parts[-1]
        target_dict[last_key] = var_array

    return params

