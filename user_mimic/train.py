"""Track B trainer: unsloth + Qwen + LoRA SFT on user-mimic samples.

Run on a CUDA box (no Mac local path). One clean dispatch per sample shape;
canonical Sample comes from user_mimic.canonical via the on-disk JSONL.

Pipeline (single path):
    Sample -> render_chat -> tokenize_with_label_mask -> SFTTrainer.

Loss is computed only on the user's target tokens. Everything else (system,
context, the assistant-prompt scaffolding) is -100 in labels. The
`CAN_AUTO_RESPOND: <score>%\\nAUTO_RESPONSE: <target>` two-line contract
is stitched in at training time so the LM head learns the format itself.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

# Heavy ML deps imported lazily inside main() so `--help` and import-checks
# work in non-CUDA envs (this Mac).


# ---------------- canonical training-row type ----------------

@dataclass(frozen=True)
class Row:
    """Training row after label-join. Pure data; no Optionals in core."""
    messages: list[dict]   # [{"role": "system"|"user"|"assistant", "content": str}, ...]
    target: str            # the assistant reply we want to learn (already includes CAN_AUTO_RESPOND prefix)
    target_score: int      # 0..100, used only for diagnostics; gating happens in render
    project: str


# ---------------- label join + system prompt + target wrapping ----------------

# Defaults match scripts/label_samples.py heuristics so an unlabeled sample
# still gets a sensible score by length zone (no silent fallback: this is
# the documented "heuristic" branch).
LOW_THRESHOLD = 150
HIGH_THRESHOLD = 800


def _heuristic_score(target_len: int) -> int:
    if target_len <= LOW_THRESHOLD:
        frac = target_len / LOW_THRESHOLD if LOW_THRESHOLD else 0.0
        return int(round(100 - 10 * frac))
    if target_len <= HIGH_THRESHOLD:
        return 70  # middle without a Claude label: neutral mid-range
    import math
    return max(0, int(round(50 * math.exp(-0.002 * (target_len - HIGH_THRESHOLD)))))


def _sample_key(thread_id: str, target: str) -> str:
    import hashlib
    return f"{thread_id}:{hashlib.sha1(target.encode('utf-8')).hexdigest()[:12]}"


def _load_labels(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {}
    out: dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["key"]] = int(r["score"])
    return out


def _system_text(harness: str, model: str, cwd: str) -> str:
    return (
        f"This is a log from model {model} in harness {harness}, "
        f"cwd {cwd}. Respond as the user.\n"
        "Output exactly two lines:\n"
        "CAN_AUTO_RESPOND: <0-100>%\n"
        "AUTO_RESPONSE: <reply>"
    )


def _wrap_target(target: str, score: int) -> str:
    return f"CAN_AUTO_RESPOND: {score}%\nAUTO_RESPONSE: {target}"


def _coerce_role(role: str) -> str:
    # Qwen chat template only knows system/user/assistant.
    # Loader already collapsed tool_use/tool_result into user/assistant text.
    if role in ("system", "user", "assistant"):
        return role
    return "user"


def iter_rows(path: Path, labels: dict[str, int]) -> Iterator[Row]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            target = d["target"]
            key = _sample_key(d["thread_id"], target)
            score = labels.get(key, _heuristic_score(len(target)))
            sysmsg = {"role": "system", "content": _system_text(d["harness"], d["model"], d["cwd"])}
            ctx = [{"role": _coerce_role(m.get("role", "user")), "content": m.get("content", "")}
                   for m in d["context_messages"]]
            # Drop final assistant turn if present so the next-token target is the user-as-user reply.
            # Sample shape per README: context ends right before the user's message, so the assistant
            # role is the natural prefix; but the user's reply *is* a user message. We train it as the
            # assistant turn the bot must produce. The CAN_AUTO_RESPOND wrapper signals that.
            yield Row(
                messages=[sysmsg, *ctx],
                target=_wrap_target(target, score),
                target_score=score,
                project=d["project"],
            )


# ---------------- tokenization with label mask ----------------

@dataclass(frozen=True)
class TokConfig:
    max_seq_len: int
    pad_id: int


def tokenize_row(tokenizer, row: Row, cfg: TokConfig) -> dict:
    """One sample -> {input_ids, attention_mask, labels} with prompt fully masked.

    Strategy: render messages with add_generation_prompt=True (this gives the
    exact prefix the model would see at inference). Tokenize that prefix.
    Then tokenize prefix + target + eos. Labels = ids with prefix slice = -100.
    Trim from the LEFT (drop oldest context tokens) on overflow so the target
    is always intact -- matches loader.py truncation policy.
    """
    prompt_text = tokenizer.apply_chat_template(
        row.messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(row.target, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    target_ids = target_ids + [eos]

    # Left-truncate the prompt if total > max_seq_len. We MUST keep all target
    # tokens (loss signal) and the end of the prompt (assistant scaffolding).
    budget = cfg.max_seq_len - len(target_ids)
    if budget < 16:
        # Pathological: target alone exceeds max_seq_len. Drop loudly.
        raise ValueError(f"target too long for max_seq_len={cfg.max_seq_len}: {len(target_ids)} tokens")
    if len(prompt_ids) > budget:
        prompt_ids = prompt_ids[-budget:]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + list(target_ids)
    attention_mask = [1] * len(input_ids)

    # Right-pad to max_seq_len so the SFTTrainer collator is a no-op (we're already aligned).
    pad_n = cfg.max_seq_len - len(input_ids)
    if pad_n > 0:
        input_ids = input_ids + [cfg.pad_id] * pad_n
        labels = labels + [-100] * pad_n
        attention_mask = attention_mask + [0] * pad_n

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------- training entry point ----------------

@dataclass(frozen=True)
class TrainArgs:
    train_path: Path
    val_path: Path
    labels_path: Path | None
    output_dir: Path
    model_name: str
    max_seq_len: int
    lora_r: int
    lora_alpha: int
    epochs: int
    per_device_batch: int
    grad_accum: int
    lr: float
    warmup_ratio: float
    seed: int
    save_merged: bool


def _parse() -> TrainArgs:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="data/samples/train.jsonl", type=Path)
    p.add_argument("--val", default="data/samples/val.jsonl", type=Path)
    p.add_argument("--labels", default="data/samples/labels.jsonl", type=Path,
                   help="Optional labels jsonl; if missing, falls back to length-zone heuristic.")
    p.add_argument("--output-dir", default="out/user_mimic_lora", type=Path)
    # Qwen2.5-7B-Instruct chosen for v1: short targets don't need 30B; long context (16k)
    # is the actual constraint. Swap to "unsloth/Qwen3-30B-A3B-bnb-4bit" for the big run.
    p.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    p.add_argument("--max-seq-len", type=int, default=16384)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per-device-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--save-merged", action="store_true",
                   help="Also save merged FP16 alongside the LoRA adapter.")
    a = p.parse_args()
    return TrainArgs(
        train_path=a.train, val_path=a.val,
        labels_path=a.labels if a.labels.exists() else None,
        output_dir=a.output_dir, model_name=a.model, max_seq_len=a.max_seq_len,
        lora_r=a.lora_r, lora_alpha=a.lora_alpha, epochs=a.epochs,
        per_device_batch=a.per_device_batch, grad_accum=a.grad_accum,
        lr=a.lr, warmup_ratio=a.warmup_ratio, seed=a.seed,
        save_merged=a.save_merged,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse()

    # Heavy imports: only on the cloud box.
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    import torch

    log.info("loading model %s @ max_seq_len=%d", args.model_name, args.max_seq_len)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_len,
        dtype=None,            # unsloth picks bf16 on Ampere+
        load_in_4bit=True,
    )

    # Pad token: Qwen tokenizers don't always set one. Use eos_token as pad
    # (no silent fallback -- explicit and logged).
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("set pad_token = eos_token (id=%d)", tokenizer.pad_token_id)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_len,
    )

    labels = _load_labels(args.labels_path)
    log.info("loaded %d labels (%s)", len(labels), args.labels_path)

    cfg = TokConfig(max_seq_len=args.max_seq_len, pad_id=tokenizer.pad_token_id)

    def build_split(path: Path, name: str) -> "Dataset":
        rows = list(iter_rows(path, labels))
        log.info("%s: %d rows", name, len(rows))
        def gen():
            for r in rows:
                try:
                    yield tokenize_row(tokenizer, r, cfg)
                except ValueError as e:
                    log.warning("skip row in %s: %s", name, e)
        return Dataset.from_generator(gen)

    train_ds = build_split(args.train_path, "train")
    val_ds = build_split(args.val_path, "val")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_cfg = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
        seed=args.seed,
        max_seq_length=args.max_seq_len,
        packing=False,                 # we already padded; don't re-pack across rows
        dataset_text_field=None,       # we provide pre-tokenized rows
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_cfg,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))

    if args.save_merged:
        merged_dir = args.output_dir / "merged_fp16"
        merged_dir.mkdir(parents=True, exist_ok=True)
        log.info("saving merged FP16 to %s", merged_dir)
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")


if __name__ == "__main__":
    main()
