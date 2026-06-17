"""One-command deploy to Hugging Face Spaces."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYNC_ITEMS = [
    "mint_atlas",
    "frontend",
    "tests",
    "scripts",
    "docs",
    "data",
    "Dockerfile",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    ".dockerignore",
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy Mint Atlas to Hugging Face Spaces")
    parser.add_argument("username", help="Your Hugging Face username")
    parser.add_argument("--space", default="mint-atlas", help="Space name (default: mint-atlas)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (or set HF_TOKEN)")
    args = parser.parse_args()

    if not args.token:
        print("Error: HF token required. Set HF_TOKEN or pass --token")
        print("Get one at https://huggingface.co/settings/tokens")
        sys.exit(1)

    run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub[cli]"])

    env = os.environ.copy()
    env["HF_TOKEN"] = args.token

    repo_id = f"{args.username}/{args.space}"
    space_dir = ROOT / ".hf_deploy"

    run(
        [sys.executable, "-m", "huggingface_hub.cli.hf", "repo", "create", args.space, "--type", "space", "--space-sdk", "docker", "-y"],
        cwd=ROOT,
    )

    if space_dir.exists():
        shutil.rmtree(space_dir)

    run(["git", "clone", f"https://huggingface.co/spaces/{repo_id}", str(space_dir)])

    for item in SYNC_ITEMS:
        src = ROOT / item
        dst = space_dir / item
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        elif src.exists():
            shutil.copy2(src, dst)

    run(["git", "add", "-A"], cwd=space_dir)
    run(["git", "commit", "-m", "Deploy Mint Atlas"], cwd=space_dir)
    run(["git", "push"], cwd=space_dir)

    print(f"\nDeployed: https://huggingface.co/spaces/{repo_id}")
    print("Build takes ~5-10 min on first deploy.")


if __name__ == "__main__":
    main()
