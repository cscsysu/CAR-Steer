"""Minimal JSONL inference entry point; no benchmark labels are consumed."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from .config import SteeringConfig
from .generation import CARSteer


def main():
    parser = argparse.ArgumentParser(description="CAR-Steer core greedy inference")
    parser.add_argument("--model", required=True, help="HF model ID or local weight directory")
    parser.add_argument("--revision", default="main", help="HF revision; use a commit for reproducibility")
    settings = parser.add_mutually_exclusive_group(required=True)
    settings.add_argument("--preset", choices=["qwen3-8b", "llama-3.1-8b", "mistral-7b"])
    settings.add_argument("--config", type=Path, help="JSON SteeringConfig")
    parser.add_argument("--input", type=Path, required=True, help="JSONL with context, question, optional id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    config = (SteeringConfig.preset(args.preset) if args.preset else
              SteeringConfig(**json.loads(args.config.read_text(encoding="utf-8"))))
    rows = []
    for line_num, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("context"), str) or not isinstance(row.get("question"), str):
            parser.error(f"Input line {line_num} needs string context and question fields")
        rows.append(row)
    set_seed(42)
    dtype = getattr(torch, args.dtype or ("bfloat16" if args.device.startswith("cuda") else "float32"))
    load = dict(revision=args.revision, local_files_only=args.local_files_only, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, **load)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                                attn_implementation="eager", **load).to(args.device)
    runner = CARSteer(model, tokenizer, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            result = runner.generate(row["context"], row["question"])
            record = {"id": row.get("id", index), "config": config.to_dict(), **result.to_dict()}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
