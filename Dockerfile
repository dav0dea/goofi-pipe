# goofi-pipe, built entirely inside the container — the host needs neither Rust nor uv nor Node.
#
#   docker build -t goofi .
#   docker run --rm -it -p 8000:8000 -v .:/workdir -v goofi-home:/home/goofi goofi
#
# That run line is LITERAL: no `$HOME`, no `$(id -u)`, no `~`, no per-OS path shape. It is the
# same text in bash, zsh, fish, PowerShell and cmd, which is the point — Docker resolves `.`
# itself, and a named volume is keyed by name rather than by any host path.
#
# **Single stage on purpose.** goofi bakes three ABSOLUTE paths into the binary at compile time:
# the pyo3 RUNPATH, `option_env!("PYO3_PYTHON")`, and `env!("CARGO_MANIFEST_DIR")` behind
# `--subproc-python`. Getting any of them wrong is SILENT — goofi still starts, still serves, and
# still runs every Python node, just on the slow subprocess tier with no error anywhere. A
# multi-stage build would need a copy list enumerating everything those paths reach, where an
# omission defaults to *dropped*. Here the build tree never moves, so there is nothing to get
# wrong, and the prune below is a list of known build-only paths where an omission costs disk
# rather than correctness.

FROM rust:1.89-slim-bookworm

# `clang`/`libclang-dev` are not optional: iceoryx2-pal-posix and iceoryx2-pal-os-api run bindgen
# in their build scripts, and without libclang the failure reads like a broken machine rather than
# a missing package. Nothing in the tree pulls openssl, pkg-config or a TLS backend, so the list
# ends here. `tini` reaps — goofi is PID 1 and spawns PTY and subprocess children.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential clang libclang-dev curl ca-certificates git tini \
 && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

# uv as a pinned binary rather than a piped install script: the same version every build.
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /usr/local/bin/uv

# The agent harnesses. Only the BINARIES are baked — credentials arrive at run time through
# `$HOME`, so adding a harness here never involves a mount decision.
RUN npm install -g --no-fund --no-audit \
        @anthropic-ai/claude-code @openai/codex opencode-ai \
 && npm cache clean --force

# Both interpreters uv-managed, under one prefix. Without `only-managed` the GIL venv's base
# would be whatever python the base image happens to ship — not a thing to leave to chance in an
# image whose whole purpose is predictability.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_PREFERENCE=only-managed \
    CARGO_TERM_COLOR=always

WORKDIR /opt/goofi
COPY . .

# One RUN, because the prune has to land in the same layer as the build: a later `rm` leaves the
# deleted bytes in the image regardless.
#
# `goofi-init` BEFORE `cargo build` — it writes `.cargo/config.toml`, and cargo reads that file
# exactly once, at startup. That ordering is the whole reason goofi-init is a crate rather than a
# build script.
#
# `npm install`, never `npm ci` — `frontend/.gitignore` ships no lockfile on purpose ("a recipe,
# not a frozen build"), and `npm ci` requires one.
RUN cargo run -p goofi-init \
 && cd frontend && npm install --no-fund --no-audit && npm run build && cd .. \
 && GOOFI_SKIP_FRONTEND_BUILD=1 cargo build --release \
 && cp target/release/goofi-pipe /usr/local/bin/goofi-pipe \
 && rm -rf target frontend/node_modules "$CARGO_HOME/registry" "$CARGO_HOME/git"

# uid 1000 is the common single-user Linux desktop, so bind-mounted files land owned by the user
# instead of by root. Docker Desktop maps ownership itself on macOS and Windows, where this is
# simply inert. A host on a different uid passes `--user` and everything still works: all three
# harnesses were measured running fine with no `/etc/passwd` entry at all.
RUN groupadd -g 1000 goofi \
 && useradd -u 1000 -g 1000 -m -d /home/goofi goofi \
 && mkdir -p /workdir && chown goofi:goofi /workdir
USER goofi

# The two cwd-relative inputs move onto explicit seams, which is what frees the working directory
# to BE the user's mount. `/workdir` is the word goofi's own Save/Load modal already shows for it.
ENV GOOFI_FRONTEND_BUILD=/opt/goofi/frontend/build \
    HOME=/home/goofi
WORKDIR /workdir

EXPOSE 8000

# No CMD: Docker APPENDS run arguments to ENTRYPOINT, so `docker run goofi --port 9000` works.
# The two baked flags compose by different rules, each the right one — `--bind` is last-wins, so a
# user's supersedes this default; `--auto-nodes` accumulates, so a user's directory is ADDED to the
# builtin tree rather than replacing it.
ENTRYPOINT ["/usr/bin/tini", "--", "goofi-pipe", "--bind", "0.0.0.0", "--auto-nodes", "/opt/goofi/nodes"]
