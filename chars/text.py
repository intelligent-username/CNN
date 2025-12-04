"""Keep track of text order for reconstructing latere"""

import torch
from torch.nn.utils.rnn import pad_sequence

def tokenize_text(text):
    """
    Convert string into a list of integer token IDs.
    Replace this with your vocabulary mapping.
    """
    return [ord(c) for c in text]  # naive example, one char -> one int

def collate_fn(crops, tokenized_targets, device):
    """
    Pads sequences to the same length and stacks images.
    crops: list of tensors [C,H,W]
    tokenized_targets: list of lists of ints
    """
    batch_inputs = torch.stack(crops).to(device)
    targets = [torch.tensor(t, dtype=torch.long) for t in tokenized_targets]
    batch_targets = pad_sequence(targets, batch_first=True, padding_value=0).to(device)
    return batch_inputs, batch_targets



