#!/usr/bin/env bash
# Install (or update) audio_input on a Mac with Apple Silicon:
#
#   curl -fsSL https://raw.githubusercontent.com/Frielon/audio_input/main/install.sh | bash
#
# Everything except the models goes into one folder, ~/audio_input:
#   the code   a git clone (without git: the archive of the main branch)
#   .uv/       uv (a small Python package manager), the Python 3.11 it downloads and its cache
#   .venv/     the packages of requirements.txt (about 2 GB)
# The models (about 6 GB) land in ~/.cache/huggingface while a sentence spoken by `say` is
# analysed, which also shows that everything works.  Nothing else on the Mac is changed.
#
# Run it again to update.  Run as ./install.sh in a clone, it sets up that clone as it is.
#   AUDIO_INPUT_DIR    install folder (curl … | AUDIO_INPUT_DIR=~/apps/audio_input bash)
#   AUDIO_INPUT_REPO   GitHub repository to install from (default Frielon/audio_input)
set -euo pipefail

REPO="${AUDIO_INPUT_REPO:-Frielon/audio_input}"
SAMPLE="Yesterday I meet my advisor, and he say we should use calculator."
CHECKOUT=0  # 1: run from a clone, which is used as it is
if [[ -n "${AUDIO_INPUT_DIR:-}" ]]; then
  INSTALL_DIR="$AUDIO_INPUT_DIR"
elif [[ -f "${BASH_SOURCE[0]:-}" && -f "$(dirname "${BASH_SOURCE[0]}")/audio_input/__main__.py" ]]; then
  INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  CHECKOUT=1
else
  INSTALL_DIR="$HOME/audio_input"
fi
case "$INSTALL_DIR" in /*) ;; *) INSTALL_DIR="$PWD/$INSTALL_DIR" ;; esac
VENV="$INSTALL_DIR/.venv"
UV="$INSTALL_DIR/.uv/bin/uv"
# uv's Python and download cache stay in the install folder too: deleting the folder removes everything
export UV_PYTHON_INSTALL_DIR="$INSTALL_DIR/.uv/python" UV_CACHE_DIR="$INSTALL_DIR/.uv/cache"

step() { printf '\n\033[1;34m==>\033[0;1m %s\033[0m\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

check_mac() {
  [[ "$(uname -s)" == Darwin ]] || die "audio_input runs on macOS only."
  [[ "$(sysctl -n hw.optional.arm64 2>/dev/null)" == 1 ]] || die "audio_input needs a Mac with Apple Silicon (M1 or newer)."
  [[ "$(uname -m)" == arm64 ]] || die "this terminal runs under Rosetta (x86_64); open a native one and try again."
  # espeak-ng (the IPA) reads its data from inside .venv, and not from a path of 160 characters or more
  local data="$VENV/lib/python3.11/site-packages/espeakng_loader/espeak-ng-data"
  if (( ${#data} >= 160 )) && [[ -z "${ESPEAK_DATA_PATH:-}" ]]; then
    die "the path $INSTALL_DIR is too long for espeak-ng; choose a folder whose path has at most $(( 159 - ${#data} + ${#INSTALL_DIR} )) characters (AUDIO_INPUT_DIR)."
  fi
  local free_gb
  free_gb="$(df -g "$HOME" | awk 'NR == 2 {print $4}')"
  if [[ "$free_gb" =~ ^[0-9]+$ ]] && (( free_gb < 12 )); then
    echo "warning: ${free_gb} GB free on disk; the packages and the models take about 8 GB." >&2
  fi
}

have_git() {  # without the Xcode Command Line Tools, /usr/bin/git only offers to install them
  local git_path
  git_path="$(command -v git)" || return 1
  [[ "$git_path" != /usr/bin/git ]] || xcode-select -p >/dev/null 2>&1
}

get_code() {
  if (( CHECKOUT )); then
    step "Setting up the clone in $INSTALL_DIR"
  elif [[ -d "$INSTALL_DIR/.git" ]]; then
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

get_uv() {
  [[ -x "$UV" ]] && return
  step "Downloading uv, the Python package manager (about 30 MB)"
  mkdir -p "$(dirname "$UV")"
  curl -fsSL https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz |
    tar -xz --strip-components 1 -C "$(dirname "$UV")"
}

make_env() {
  if [[ ! -x "$VENV/bin/python" ]]; then
    step "Creating the Python 3.11 environment in $VENV"
    "$UV" venv --clear --managed-python --python 3.11 "$VENV"
  fi
  step "Installing the packages of requirements.txt (about 2 GB the first time)"
  "$UV" pip install --python "$VENV/bin/python" -r "$INSTALL_DIR/requirements.txt"
}

try_it() {
  step "Downloading the models (about 6 GB, only the first time) and trying them on a sentence spoken by say"
  local tmp
  tmp="$(mktemp -d)"
  say -v Samantha -o "$tmp/sample.aiff" "$SAMPLE" 2>/dev/null || say -o "$tmp/sample.aiff" "$SAMPLE"
  "$INSTALL_DIR/run.sh" --file "$tmp/sample.aiff"
  rm -rf "$tmp"
}

done_msg() {
  local dir="$INSTALL_DIR"
  case "$dir" in "$HOME"/*) dir="~${dir#"$HOME"}" ;; esac
  step "Done: audio_input is installed in $dir"
  cat <<EOF

Start it from a Terminal window you keep open while you use it:

    $dir/run.sh

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
  get_uv
  make_env
  try_it
  done_msg
}

# Nothing above has an effect before this line, so a download cut short does nothing; under
# `curl | bash` stdin is the script itself, so no step may read it.
main "$@" </dev/null
