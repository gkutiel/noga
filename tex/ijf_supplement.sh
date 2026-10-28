#!/usr/bin/env bash
# Build the anonymised code-and-data supplement for the IJF submission:
# tex/ijf_supplement.zip with the code, the raw sources and data/data.csv.
# Fails if any file in it names the authors.
set -euo pipefail
cd "$(dirname "$0")/.."

out=tex/ijf_supplement.zip
files=(
    README.md CLAUDE.md Taskfile.yml pyproject.toml uv.lock .python-version
    noga/*.py
    data/noga.xlsx data/ims.he.csv data/Smp_01_04_2023-01_01_2026_he-IL.xlsx
    data/data.csv
)

if grep -l -i -E "kutiel|aleksandrowicz|avigal|technion|gilad" "${files[@]}"; then
    echo "identifying text in the files above" >&2
    exit 1
fi

rm -f "$out"
# -X: no extra file attributes (owner ids) in the archive.
zip -q -X "$out" "${files[@]}"
ls -la "$out"
