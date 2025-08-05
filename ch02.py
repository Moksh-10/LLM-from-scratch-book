import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken

class dataset(Dataset):
    def __init__(self, txt, tokenizer, max_len, stride):
        super().__init__()
        self.tok = tokenizer
        self.input_ids = []
        self.target_ids = []

        tids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(tids) - max_len, stride):
            inp = tids[i:i+max_len]
            tar = tids[i+1:i+1+max_len]
            self.input_ids.append(torch.tensor(inp))
            self.target_ids.append(torch.tensor(tar))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def dataloader(txt, batch_size=4, max_len=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tok = tiktoken.get_encoding("gpt2")
    ds = dataset(txt, tok, max_len, stride)
    dl = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dl