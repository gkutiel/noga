#!/usr/bin/env bash
# Build the anonymised code-and-data supplement for the IJF submission:
# tex/ijf_supplement.zip with the code, data/data.csv and its own README.
# Fails if any file in it names the authors.
set -euo pipefail
cd "$(dirname "$0")/.."

out=tex/ijf_supplement.zip
files=(
    pyproject.toml uv.lock .python-version
    noga/*.py
    data/data.csv
)

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cp --parents "${files[@]}" "$stage"
cp tex/ijf_supplement_README.md "$stage/README.md"
# report.py writes its figures to tex/plot/.
mkdir -p "$stage/tex/plot"

cd "$stage"
if grep -r -l -i -E "kutiel|aleksandrowicz|avigal|technion|gilad" .; then
    echo "identifying text in the files above" >&2
    exit 1
fi

rm -f "$OLDPWD/$out"
# -X: no extra file attributes (owner ids) in the archive.
zip -q -r -X "$OLDPWD/$out" .
cd "$OLDPWD"
ls -la "$out"
