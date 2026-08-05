#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"

profile="cpu"
requirements="requirements.txt"
if [[ "${1:-}" == "--gpu" ]]; then
  profile="gpu"
  requirements="requirements-gpu.txt"
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--gpu]" >&2
  exit 2
fi

build_venv="$project_dir/.package-${profile}-venv"
if [[ ! -x "$build_venv/bin/python" ]]; then
  .venv/bin/python -m venv "$build_venv"
fi
"$build_venv/bin/pip" install -r "$requirements" pyinstaller

"$build_venv/bin/pyinstaller" --noconfirm --clean packaging/swapio.spec
echo "Swapio ${profile^^} bundle: $project_dir/dist/swapio/"
