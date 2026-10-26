'''
Trains one model from a clean working tree, so every run maps to a commit.

    uv run train <model> [--seed N] [--price-seed N] [--force]

<model> is a short name: mse, l1, med, opt, p95, cost.
'''
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from noga.cost import SHORT
from noga.model import git_sha, pt, train

BY_SHORT = {short: name for name, short in SHORT.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("model", choices=list(BY_SHORT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--price-seed", type=int, default=0)
    parser.add_argument(
        "--force", action="store_true",
        help="retrain even if this commit's model exists (the old file "
             "moves to models/old/)")
    args = parser.parse_args()

    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True).stdout
    if dirty:
        sys.exit("Refusing to train: commit or stash these changes first\n"
                 + dirty)

    name = BY_SHORT[args.model]
    out = pt.model(name, args.seed, args.price_seed, sha=git_sha())
    if args.force and out.exists():
        old = Path("models/old")
        old.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        moved = out.rename(old / f"{out.stem}-{stamp}{out.suffix}")
        print(f"Moved {out} to {moved}")

    train(name, seed=args.seed, price_seed=args.price_seed, force=args.force)


if __name__ == "__main__":
    main()
