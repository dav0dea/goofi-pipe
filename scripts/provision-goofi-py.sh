#!/usr/bin/env bash
# Provision the `goofi` Python package (the goofi-pymod crate, built as a wheel) into the
# repo's two machine-local, gitignored venvs — so both Python node tiers work:
#
#   .ftvenv  (free-threaded 3.14t)  the in-process host's introspection PROBE spawns
#                                    `python -c "import goofi; goofi.introspect(...)"`, so
#                                    the FT interpreter must have goofi importable.
#   .venv    (a GIL Python >= 3.8)   the SUBPROCESS child runs `import goofi; goofi.serve()`.
#
# Both venvs are provisioned separately (build.rs makes .ftvenv for pyo3; make .venv with
# `uv venv --python 3.10 .venv && uv pip install --python .venv/bin/python numpy`). This
# script only installs goofi into whichever of them exist. Re-run after changing goofi-pymod.
#
# Why a script and not build.rs: building the wheel runs maturin, which invokes cargo — a
# nested cargo under a cargo build script would deadlock on the target-dir build lock. So the
# wheel build lives here; build.rs only *checks* that goofi is importable and points here.
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root

WHEELS="target/wheels"
PYMOD="crates/goofi-pymod/Cargo.toml"
mkdir -p "$WHEELS"

install_into() {
    # $1 = venv python, $2 = wheel-name glob (distinguishes the FT-native vs abi3 wheel)
    local py="$1" glob="$2"
    [ -x "$py" ] || { echo "skip: $py not found"; return; }
    echo ">> building + installing goofi into $py"
    maturin build --release -i "$py" -o "$WHEELS" -m "$PYMOD"
    # shellcheck disable=SC2086
    uv pip install --python "$py" --force-reinstall --no-deps $WHEELS/$glob
}

# Free-threaded build: maturin ignores abi3 for a free-threaded interpreter and emits a
# native cp314t wheel; the GIL build emits a single cp38-abi3 wheel usable by any GIL >= 3.8.
install_into ".ftvenv/bin/python" "goofi-*cp3*t-*.whl"
install_into ".venv/bin/python"   "goofi-*abi3*.whl"

echo "done — goofi provisioned into the present venvs"
