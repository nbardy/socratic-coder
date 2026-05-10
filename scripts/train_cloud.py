"""STUB: cloud training launcher for Track B.

THIS SCRIPT IS NOT YET WIRED. It is the documented skeleton of the steps a
human (or a follow-up agent) needs to run to train user_mimic.train on a
remote CUDA box. The actual cloud-API calls are intentionally commented
out so this file is safe to import / lint locally.

Pick ONE provider before turning this on:
  - runpod   (cheapest A100/H100 on-demand; SSH + pod-template flow)
  - modal    (containers; Python SDK; nice for reproducible runs)
  - lambdalabs / fluidstack / vast.ai (cheapest spot, flakier)

Workflow (all providers share this shape):
  1. Provision   -- create a GPU box (1x A100-80G or 1x H100 sxm).
  2. Push code   -- rsync repo (omit data/raw/, .venv/, out/).
  3. Push data   -- rsync data/samples/{train,val}.jsonl + labels.jsonl.
  4. Bootstrap   -- install uv on the box, `uv sync --extra train`.
  5. Train       -- `uv run python -m user_mimic.train --model ... --output-dir out/...`.
  6. Pull        -- rsync out/<run_id>/ back to local.
  7. Teardown    -- destroy pod (don't pay for idle GPU).

Step 4 install command on the cloud box:
    curl -LsSf https://astral.sh/uv/install.sh | sh
    cd user_mimic && uv sync --extra train

Step 5 example invocation:
    uv run python -m user_mimic.train \\
        --train data/samples/train.jsonl \\
        --val   data/samples/val.jsonl \\
        --labels data/samples/labels.jsonl \\
        --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \\
        --max-seq-len 16384 \\
        --epochs 1 \\
        --output-dir out/qwen7b_r32_e1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CloudArgs:
    provider: str           # "runpod" | "modal" | "lambdalabs"
    gpu: str                # "A100-80G" | "H100-80G"
    model: str              # HF repo id or unsloth bnb tag
    run_name: str           # output dir slug, e.g. "qwen7b_r32_e1"
    repo_dir: Path          # local user_mimic checkout
    remote_user_host: str   # "root@1.2.3.4" once pod is up


def _parse() -> CloudArgs:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", default="runpod", choices=["runpod", "modal", "lambdalabs"])
    p.add_argument("--gpu", default="A100-80G")
    p.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    p.add_argument("--run-name", default="qwen7b_r32_e1")
    p.add_argument("--repo-dir", type=Path, default=Path.cwd())
    p.add_argument("--remote", default="",
                   help="SSH target like root@1.2.3.4. Empty = print plan only.")
    a = p.parse_args()
    return CloudArgs(
        provider=a.provider, gpu=a.gpu, model=a.model,
        run_name=a.run_name, repo_dir=a.repo_dir,
        remote_user_host=a.remote,
    )


# --- step impls (real shells; provisioning calls are stubbed) ---

def provision(args: CloudArgs) -> str:
    """Spin up a GPU pod; return ssh target. STUBBED."""
    # if args.provider == "runpod":
    #     import runpod  # pip install runpod
    #     pod = runpod.create_pod(name=args.run_name, gpu_type_id="NVIDIA A100 80GB PCIe",
    #                             image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel",
    #                             container_disk_in_gb=80, volume_in_gb=200,
    #                             ports="22/tcp")
    #     return f"root@{pod['publicIp']}"
    # if args.provider == "modal":
    #     # Modal is image-based, not SSH; rewrite this to a Modal app entrypoint.
    #     raise NotImplementedError("modal path: rewrite as @app.function(gpu='A100')")
    raise NotImplementedError("provision() stubbed -- pick provider, install SDK, fill in.")


def push_code(args: CloudArgs) -> None:
    cmd = [
        "rsync", "-az", "--delete",
        "--exclude=.venv", "--exclude=out", "--exclude=data/raw",
        "--exclude=__pycache__", "--exclude=.git",
        f"{args.repo_dir}/",
        f"{args.remote_user_host}:user_mimic/",
    ]
    print("CODE PUSH:", " ".join(cmd))
    # subprocess.run(cmd, check=True)


def push_data(args: CloudArgs) -> None:
    cmd = [
        "rsync", "-az", "--progress",
        f"{args.repo_dir}/data/samples/",
        f"{args.remote_user_host}:user_mimic/data/samples/",
    ]
    print("DATA PUSH:", " ".join(cmd))
    # subprocess.run(cmd, check=True)


def bootstrap(args: CloudArgs) -> None:
    remote_cmd = (
        "curl -LsSf https://astral.sh/uv/install.sh | sh && "
        "cd user_mimic && ~/.local/bin/uv sync --extra train"
    )
    cmd = ["ssh", args.remote_user_host, remote_cmd]
    print("BOOTSTRAP:", " ".join(cmd))
    # subprocess.run(cmd, check=True)


def train(args: CloudArgs) -> None:
    remote_cmd = (
        f"cd user_mimic && ~/.local/bin/uv run python -m user_mimic.train "
        f"--model {args.model} --output-dir out/{args.run_name} "
        f"--epochs 1 --save-merged"
    )
    cmd = ["ssh", args.remote_user_host, remote_cmd]
    print("TRAIN:", " ".join(cmd))
    # subprocess.run(cmd, check=True)


def pull_artifacts(args: CloudArgs) -> None:
    cmd = [
        "rsync", "-az", "--progress",
        f"{args.remote_user_host}:user_mimic/out/{args.run_name}/",
        f"{args.repo_dir}/out/{args.run_name}/",
    ]
    print("PULL:", " ".join(cmd))
    # subprocess.run(cmd, check=True)


def teardown(args: CloudArgs) -> None:
    """Destroy pod. STUBBED -- provider-specific."""
    print("TEARDOWN: (stubbed; destroy the pod manually or wire the SDK call here)")


# --- thin dispatcher ---

def main() -> None:
    args = _parse()
    if not args.remote_user_host:
        print(__doc__)
        print(f"\nplanned run: provider={args.provider} gpu={args.gpu} model={args.model} run={args.run_name}")
        print("re-run with --remote root@<ip> to dry-print the rsync/ssh commands.")
        return
    push_code(args)
    push_data(args)
    bootstrap(args)
    train(args)
    pull_artifacts(args)


if __name__ == "__main__":
    main()
