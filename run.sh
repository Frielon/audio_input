#!/usr/bin/env bash
# Start the dictation app (menu-bar icon + floating overlay).
#   ./run.sh                       -> hold the left Option key and speak
#   ./run.sh --hotkey f13          -> any option of `python -m audio_input --help`
set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME="${CONDA_ENV_NAME:-audio-input}"

if [[ "$(basename "${CONDA_PREFIX:-}")" != "$ENV_NAME" ]]; then
  # conda from the shell, else where install.sh (Miniforge) or the usual installers put it
  for conda_exe in "${CONDA_EXE:-}" "$(command -v conda || true)" \
                   "$HOME"/{miniforge3,mambaforge,miniconda3,anaconda3}/bin/conda \
                   /opt/homebrew/Caskroom/{miniforge,miniconda}/base/bin/conda /opt/{miniconda3,anaconda3}/bin/conda; do
    [[ -n "$conda_exe" && -x "$conda_exe" ]] && break
  done
  if [[ ! -x "$conda_exe" ]]; then
    echo "conda not found; install everything with:" >&2
    echo "  curl -fsSL https://raw.githubusercontent.com/Frielon/audio_input/main/install.sh | bash" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  eval "$("$conda_exe" shell.bash hook)"
  conda activate "$ENV_NAME"
fi

exec python -m audio_input "$@"
