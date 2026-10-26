'''
Trains one model from a clean working tree, so every run maps to a commit.

    uv run train <model> [--seed N] [--price-seed N]

<model> is a short name: mse, l1, med, opt, p95, cost.
'''
import argparse
import subprocess
import sys

from noga.cost import SHORT
from noga.model import train

BY_SHORT = {short: name for name, short in SHORT.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("model", choices=list(BY_SHORT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--price-seed", type=int, default=0)
    args = parser.parse_args()

    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True).stdout
    if dirty:
        sys.exit("Refusing to train: commit or stash these changes first\n"
                 + dirty)

    train(BY_SHORT[args.model], seed=args.seed, price_seed=args.price_seed)


if __name__ == "__main__":
    main()
