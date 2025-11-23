# import torch
# from torch import nn
# import os
# from pathlib import Path
# import tiktoken
# from tiktoken.load import load_tiktoken_bpe
#
#
# LLAMA32_CONFIG_1B = {
#     "vocab_size": 128_256,
#     "context_length": 131_072,
#     "emb_dim": 2048,
#     "n_heads": 32,
#     "n_layers": 16,
#     "hidden_dim": 8192,
#     "n_kv_groups": 8,
#     "rope_base": 500_000.0,
#     "dtype": torch.bfloat16,
#     "rope_freq": {
#         "factor": 32.0,
#         "low_freq_factor": 1.0,
#         "high_freq_factor": 4.0,
#         "original_context_length": 8192,
#     }
# }
#
#
# def apply_rope(x, cos, sin):
#     bs, nh, sl, hd = x.shape
#     assert hd % 2 == 0, "head dim must be even"
#
#     x1 = x[..., : hd // 2]
#     x2 = x[..., hd // 2:]
#
#     cos = cos[:sl, :].unsquueze(0).unsqueeze(0)
#     sin = sin[:sl, :].unsqueeze(0).unsqueeze(0)
#
#     rot = torch.cat((-x2, x1), dim=-1)
#     x_rot = (x * cos) + (rot * sin)
#
#     return x_rot.to(dtype=x.dtype)
#
#
