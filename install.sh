#!/usr/bin/env bash
# Install (or update) audio_input on a Mac with Apple Silicon:
#
#   curl -fsSL https://raw.githubusercontent.com/Frielon/audio_input/main/install.sh | bash
#
# 1. the code, in ~/audio_input (git clone; without git, the archive of the main branch)
# 2. conda: when there is none, Miniforge in ~/miniforge3 (shell settings are left alone)
# 3. the conda env of environment.yml
# 4. the models (about 6 GB, once): a sentence spoken by `say` is analysed, which also shows that it all works
#
# Run it again to update.  Options go before `bash` (curl … | AUDIO_INPUT_DIR=~/apps/audio_input bash):
#   AUDIO_INPUT_DIR    install folder (default ~/audio_input)
#   CONDA_ENV_NAME     conda env (default audio-input; run.sh reads it too)
#   AUDIO_INPUT_REPO   GitHub repository to install from (default Frielon/audio_input)
set -euo pipefail

REPO="${AUDIO_INPUT_REPO:-Frielon/audio_input}"
INSTALL_DIR="${AUDIO_INPUT_DIR:-$HOME/audio_input}"
ENV_NAME="${CONDA_ENV_NAME:-audio-input}"
SAMPLE="Yesterday I meet my advisor, and he say we should use calculator."

step() { printf '\n\033[1;34m==>\033[0;1m %s\033[0m\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

check_mac() {
  [[ "$(uname -s)" == Darwin ]] || die "audio_input runs on macOS only."
  [[ "$(sysctl -n hw.optional.arm64 2>/dev/null)" == 1 ]] || die "audio_input needs a Mac with Apple Silicon (M1 or newer)."
  [[ "$(uname -m)" == arm64 ]] || die "this terminal runs under Rosetta (x86_64); open a native one and try again."
  local free_gb
  free_gb="$(df -g "$HOME" | awk 'NR == 2 {print $4}')"
  if [[ "$free_gb" =~ ^[0-9]+$ ]] && (( free_gb < 12 )); then
    echo "warning: ${free_gb} GB free on disk; the env and the models take about 10 GB." >&2
  fi
}

have_git() {  # without the Xcode Command Line Tools, /usr/bin/git only offers to install them
  local git_path
  git_path="$(command -v git)" || return 1
  [[ "$git_path" != /usr/bin/git ]] || xcode-select -p >/dev/null 2>&1
}

get_code() {
  if [[ -d "$INSTALL_DIR/.git" ]]; then
    step "Updating the code in $INSTALL_DIR"
    git -C "$INSTALL_DIR" pull --ff-only
  elif [[ -n "$(ls -A "$INSTALL_DIR" 2>/dev/null)" && ! -f "$INSTALL_DIR/audio_input/__main__.py" ]]; then
    die "$INSTALL_DIR exists and is not audio_input: move it away, or choose another folder with AUDIO_INPUT_DIR."
  elif have_git && [[ -z "$(ls -A "$INSTALL_DIR" 2>/dev/null)" ]]; then
    step "Downloading the code into $INSTALL_DIR"
    git clone "https://github.com/$REPO.git" "$INSTALL_DIR"
  else  # no git, or an earlier install without it: unpack the latest code over it (config.toml and data/ stay)
    step "Downloading the code into $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
    curl -fsSL "https://github.com/$REPO/archive/refs/heads/main.tar.gz" | tar -xz --strip-components 1 -C "$INSTALL_DIR"
  fi
}

find_conda() {  # the same places run.sh looks
  local c
  for c in "${CONDA_EXE:-}" "$(command -v conda || true)" \
           "$HOME"/{miniforge3,mambaforge,miniconda3,anaconda3}/bin/conda \
           /opt/homebrew/Caskroom/{miniforge,miniconda}/base/bin/conda /opt/{miniconda3,anaconda3}/bin/conda; do
    if [[ -n "$c" && -x "$c" ]]; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

get_conda() {
  if CONDA="$(find_conda)"; then
    step "Using conda at $CONDA"
    return
  fi
  step "Installing Miniforge (conda) into ~/miniforge3"
  local tmp
  tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/miniforge.sh" \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
  bash "$tmp/miniforge.sh" -b -u -p "$HOME/miniforge3"
  rm -rf "$tmp"
  CONDA="$HOME/miniforge3/bin/conda"
}

make_env() {
  cd "$INSTALL_DIR"
  if "$CONDA" env list | awk -v name="$ENV_NAME" '$1 == name {found = 1} END {exit !found}'; then
    step "Updating the conda env $ENV_NAME"
    "$CONDA" env update -n "$ENV_NAME" -f environment.yml
  else
    step "Creating the conda env $ENV_NAME (Python 3.11 and about 2 GB of packages)"
    "$CONDA" env create -n "$ENV_NAME" -f environment.yml
  fi
}

try_it() {
  step "Downloading the models (about 6 GB, only the first time) and trying them on a sentence spoken by say"
  local tmp
  tmp="$(mktemp -d)"
  say -v Samantha -o "$tmp/sample.aiff" "$SAMPLE" 2>/dev/null || say -o "$tmp/sample.aiff" "$SAMPLE"
  CONDA_EXE="$CONDA" CONDA_ENV_NAME="$ENV_NAME" "$INSTALL_DIR/run.sh" --file "$tmp/sample.aiff"
  rm -rf "$tmp"
}

done_msg() {
  local dir="$INSTALL_DIR" env_prefix=""
  case "$dir" in "$HOME"/*) dir="~${dir#"$HOME"}" ;; esac
  [[ "$ENV_NAME" == audio-input ]] || env_prefix="CONDA_ENV_NAME=$ENV_NAME "
  step "Done: audio_input is installed in $dir"
  cat <<EOF

Start it from a Terminal window you keep open while you use it:

    $env_prefix$dir/run.sh

The first time, macOS asks for permissions for the app you start it from (Terminal, iTerm …):
turn on Accessibility, Input Monitoring and Microphone in System Settings → Privacy & Security,
then quit that app (⌘Q), open it again and start audio_input again.

Then click into any text field, hold the left Option key, speak, and let go.
Settings: copy $dir/config.example.toml to config.toml.  To update, run the install command again.
EOF
}

main() {
  check_mac
  get_code
  get_conda
  make_env
  try_it
  done_msg
}

# Everything above only defines functions, so a download cut short runs nothing; stdin is the
# script itself under `curl | bash`, so no step may read it.
main "$@" </dev/null
