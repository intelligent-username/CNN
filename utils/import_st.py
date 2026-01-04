"""Ensure CaptionedSynthText is fully cached locally.

This checks whether the HuggingFace `datasets` Arrow shards exist under
`data/SynthText/raw`. If shards are missing, it triggers a download from HF.

Why this exists: an "is the folder non-empty" check is not enough; partial
downloads leave the cache populated but incomplete.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from datasets import load_dataset


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _synthtext_root() -> str:
    return os.path.join(_project_root(), "data", "SynthText", "raw")


def _find_dataset_info_json(root: str) -> Optional[str]:
    # Typical path:
    # data/SynthText/raw/wendlerc___captioned_synth_text/default/0.0.0/<hash>/dataset_info.json
    for dirpath, _, filenames in os.walk(root):
        if "dataset_info.json" in filenames:
            candidate = os.path.join(dirpath, "dataset_info.json")
            if "wendlerc___captioned_synth_text" in candidate.replace("\\", "/"):
                return candidate
    return None


def _expected_train_shards(root: str) -> int:
    info_path = _find_dataset_info_json(root)
    if info_path and os.path.isfile(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            shard_lengths = info.get("splits", {}).get("train", {}).get("shard_lengths", [])
            if isinstance(shard_lengths, list) and len(shard_lengths) > 0:
                return int(len(shard_lengths))
        except Exception:
            pass

    # Fallback: CaptionedSynthText usually materializes as 74 Arrow shards:
    # captioned_synth_text-train-00000-of-00074.arrow
    return 74


def _count_train_arrow_shards(root: str) -> int:
    pat = re.compile(r"captioned_synth_text-train-\d{5}-of-\d{5}\.arrow$", re.IGNORECASE)
    count = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if pat.search(fn):
                count += 1
    return count


def main() -> None:
    synthtext_root = _synthtext_root()
    os.makedirs(synthtext_root, exist_ok=True)

    expected = _expected_train_shards(synthtext_root)
    have = _count_train_arrow_shards(synthtext_root)

    print(f"[SynthText] Cache dir: {synthtext_root}")
    print(f"[SynthText] Arrow shards: {have}/{expected}")

    if have >= expected:
        print("[SynthText] Looks complete; skipping download.")
        return

    print("[SynthText] Missing shards; downloading from HuggingFace...")
    # This will reuse existing cache and download missing pieces when needed.
    _ = load_dataset("wendlerc/CaptionedSynthText", cache_dir=synthtext_root)

    have2 = _count_train_arrow_shards(synthtext_root)
    print(f"[SynthText] Arrow shards after download: {have2}/{expected}")
    if have2 < expected:
        raise RuntimeError(
            "Dataset cache still incomplete. "
            "Try deleting data/SynthText/raw/wendlerc___captioned_synth_text and re-running."
        )


if __name__ == "__main__":
    main()

