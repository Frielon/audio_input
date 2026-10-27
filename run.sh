#!/usr/bin/env bash
# Start the dictation app (menu-bar icon + floating overlay).
#   ./run.sh                       -> hold the left Option key and speak
#   ./run.sh --hotkey f13          -> any option of `python -m audio_input --help`
set -euo pipefail
cd "$(dirname "$0")"

if [[ -x .venv/bin/python ]]; then  # made by install.sh
  exec .venv/bin/python -m audio_input "$@"
fi

# otherwise the conda env of environment.yml
ENV_NAME="${CONDA_ENV_NAME:-audio-input}"

if [[ "$(basename "${CONDA_PREFIX:-}")" != "$ENV_NAME" ]]; then
  if [[ -x "$(command -v conda)" ]]; then
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
  else
    echo "No Python environment yet: run ./install.sh (or: conda env create -f environment.yml)" >&2
    exit 1
  fi
fi

exec python -m audio_input "$@"
