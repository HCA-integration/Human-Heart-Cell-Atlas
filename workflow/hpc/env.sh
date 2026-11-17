# Conda environments for the SLURM jobs. Source from a job running in the repository root:
#   source workflow/hpc/env.sh
#   PY=$(hhca_env atlas)/bin/python
#
# hhca_env <name> builds envs/<name>.yml into .envs/<name> on first use (micromamba) and echoes the prefix.
# micromamba is $HHCA_MAMBA if set, otherwise a pinned static binary downloaded into .tools/<version>/.
# An environment counts as built once .envs/<name>/.hhca_complete exists, so an interrupted build is redone.
# The helper package src/hhca is linked into every Python environment (hhca_link).

HHCA_MAMBA_VERSION="2.3.2-0"

# Ignore ~/.local site-packages: otherwise pip inside an env sees user installs, skips pinned packages,
# and the env silently depends on the user's home directory.
export PYTHONNOUSERSITE=1

hhca_mamba() {
  if [ -n "${HHCA_MAMBA:-}" ] && [ -x "$HHCA_MAMBA" ]; then
    echo "$HHCA_MAMBA"
    return
  fi
  # the binary must be named "micromamba": its pip: helper re-invokes it by name
  local dir="$PWD/.tools/$HHCA_MAMBA_VERSION" bin
  bin="$dir/micromamba"
  if [ ! -x "$bin" ]; then
    # jobs starting together in a fresh checkout: one downloads (lock), each into its own temp file; mv is atomic,
    # so no job executes a binary another job is still writing ("Text file busy")
    mkdir -p "$dir"
    (
      if command -v flock >/dev/null; then flock 8; fi
      if [ ! -x "$bin" ]; then
        curl -sSL --fail -o "$bin.part.$$" \
          "https://github.com/mamba-org/micromamba-releases/releases/download/$HHCA_MAMBA_VERSION/micromamba-linux-64" &&
          chmod +x "$bin.part.$$" && mv -f "$bin.part.$$" "$bin"
      fi
    ) 8>"$dir/.download.lock"
  fi
  echo "$bin"
}

# hhca_link <prefix>: make the helper package (src/hhca) importable in an environment by writing a .pth file
# into its site-packages, the link that `pip install -e . --no-deps` creates. Environments without Python
# (the R ones) are skipped.
hhca_link() {
  local python="$1/bin/python" site
  [ -x "$python" ] || return 0
  site=$("$python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') || return 1
  if [ "$(cat "$site/hhca.pth" 2>/dev/null)" != "$PWD/src" ]; then
    echo "$PWD/src" > "$site/hhca.pth.$$" && mv -f "$site/hhca.pth.$$" "$site/hhca.pth"
  fi
}

hhca_env() {
  local name="$1" prefix="$PWD/.envs/$1"
  mkdir -p "$PWD/.envs"
  (
    if command -v flock >/dev/null; then flock 9; fi
    if [ ! -f "$prefix/.hhca_complete" ]; then
      if [ -d "$prefix" ] && [ ! -f "$prefix.building" ]; then
        # no completion marker but nobody is building: reuse it, other jobs may be running from this env
        echo "hhca_env: $prefix has no completion marker; delete it to rebuild" >&2
      else
        local mamba
        mamba=$(hhca_mamba)
        touch "$prefix.building"
        rm -rf "$prefix"
        PATH="$(dirname "$mamba"):$PATH" MAMBA_ROOT_PREFIX="$PWD/.tools/mamba_root" \
          "$mamba" create -y -q -p "$prefix" -f "envs/$name.yml" >&2 || exit 1
        sha256sum "envs/$name.yml" > "$prefix/.hhca_complete"
        rm -f "$prefix.building"
      fi
    fi
    hhca_link "$prefix" >&2 || exit 1
  ) 9>"$prefix.lock" || return 1
  echo "$prefix"
}
