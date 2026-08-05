#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"

build_venv="$project_dir/.package-venv"
if [[ ! -x "$build_venv/bin/python" ]]; then
  .venv/bin/python -m venv "$build_venv"
fi
"$build_venv/bin/pip" install -r requirements.txt pyinstaller

"$build_venv/bin/pyinstaller" --noconfirm --clean packaging/swapio.spec
echo "Swapio bundle: $project_dir/dist/swapio/"
