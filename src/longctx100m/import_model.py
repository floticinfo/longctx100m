"""Import a compatible HF Llama safetensors checkpoint into this decoder."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .model import load_hf_llama_checkpoint


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Convert a plain HF Llama checkpoint into a strict longctx100m checkpoint directory"
    )
    p.add_argument("--model", required=True, help="HF repo id or local model directory")
    p.add_argument(
        "--revision", default=None, help="pin an immutable HF commit/revision"
    )
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--max-position-embeddings", type=int, default=100_002_048)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="only enable for audited model repos",
    )
    a = p.parse_args(argv)
    if a.max_position_embeddings <= 0:
        p.error("max-position-embeddings must be positive")
    if not a.revision:
        p.error(
            "pin --revision to an immutable model commit for reproducible weight/tokenizer imports"
        )
    # One importer process writes the directory; this command is deliberately not torchrun.
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "run checkpoint import once on CPU; do not launch it under torchrun"
        )
    model, tokenizer, config, report = load_hf_llama_checkpoint(
        a.model,
        revision=a.revision,
        cache_dir=a.cache_dir,
        max_position_embeddings=a.max_position_embeddings,
        trust_remote_code=a.trust_remote_code,
    )
    output = Path(a.out)
    if output.exists():
        raise ValueError(
            "output checkpoint directory already exists; choose a new path"
        )
    output.mkdir(parents=True, exist_ok=False)
    from safetensors.torch import save_file

    state = model.export_pretrained_state_dict()
    # safetensors forbids aliasing tensors; tied lm_head is reconstructed from config at load.
    if config.tie_word_embeddings and "lm_head.weight" in state:
        state.pop("lm_head.weight")
    temp = output / "model.safetensors.tmp"
    save_file(state, str(temp))
    temp.replace(output / "model.safetensors")
    (output / "config.json").write_text(
        json.dumps(config.__dict__, indent=2), encoding="utf-8"
    )
    tokenizer.save_pretrained(output / "tokenizer")
    report.update(
        {
            "attention": "exact_dense_causal_ring",
            "pretrained_for_100m_context": False,
            "long_context_training_required": True,
        }
    )
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
