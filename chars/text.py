"""Text/token utilities and batching helpers."""

from __future__ import annotations

import string
from typing import Dict, List

import torch
import torch.nn.functional as F


# Training code uses ignore_index=0, so keep 0 as padding.
PAD_ID = 0
UNK_ID = 1

# A pragmatic default charset for SynthText-like data.
_CHARS = (
    " "
    + string.ascii_letters
    + string.digits
    + string.punctuation
)

CHAR_TO_ID: Dict[str, int] = {c: i + 2 for i, c in enumerate(_CHARS)}
ID_TO_CHAR: Dict[int, str] = {i: c for c, i in CHAR_TO_ID.items()}

VOCAB_SIZE = len(_CHARS) + 2

# Must match the decoder steps used by SynthText_CRNN.
MAX_LABEL_LEN = 25


def tokenize_text(text: str, max_len: int = MAX_LABEL_LEN) -> List[int]:
    """Convert a string into a fixed-length list of token IDs."""
    text = text or ""
    ids = [CHAR_TO_ID.get(ch, UNK_ID) for ch in text]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids = ids + [PAD_ID] * (max_len - len(ids))
    return ids


def _pad_images_to_max_size(crops: List[torch.Tensor]) -> torch.Tensor:
    if not crops:
        raise ValueError("No crops provided")

    max_h = max(int(c.shape[-2]) for c in crops)
    max_w = max(int(c.shape[-1]) for c in crops)
    padded: List[torch.Tensor] = []
    for c in crops:
        # c: [C,H,W]
        pad_h = max_h - int(c.shape[-2])
        pad_w = max_w - int(c.shape[-1])
        if pad_h > 0 or pad_w > 0:
            # Pad order for 2D: (left, right, top, bottom)
            c = F.pad(c, (0, max(pad_w, 0), 0, max(pad_h, 0)), value=0.0)
        padded.append(c)
    return torch.stack(padded, dim=0)


def collate_fn(crops, tokenized_targets, device):
    """Pad variable-width images and return tensors suitable for training.

    crops: list[Tensor[C,H,W]]
    tokenized_targets: list[list[int]] fixed-length (MAX_LABEL_LEN)
    """

    batch_inputs = _pad_images_to_max_size(crops).to(device)
    batch_targets = torch.tensor(tokenized_targets, dtype=torch.long, device=device)
    return batch_inputs, batch_targets
