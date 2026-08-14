<p align="center">
<img src=https://github.com/dav0dea/goofi-pipe/assets/36135990/60fb2ba9-4124-4ca4-96e2-ae450d55596d width="150">
</p>

<h1 align="center">goofi-pipe</h1>
<h3 align="center">Generative Organic Oscillation Feedback Isomorphism Pipeline</h3>

<p align="center">
  <a href="https://github.com/dav0dea/goofi-pipe/blob/main/LICENSE"><img alt="GitHub License" src="https://img.shields.io/github/license/dav0dea/goofi-pipe"></a>
</p>

A real-time, node-based data-processing platform for biosignals (EEG, ECG, audio,
video). You build **patches** in a browser node-graph: each node ingests,
transforms, or emits `Data`; edges carry data between output and input slots. It
targets live, high-rate streams (kHz EEG, HD video) with many simultaneous viewers.

> [!IMPORTANT]
> **This branch (`rust-rewrite`) is a ground-up rewrite of the backend in Rust.**
> The Python implementation — including its ~150-node library and the node reference
> that used to fill this file — lives on the `main` branch. Nothing here is released
> yet; there is no PyPI package for this branch.
>
> **Status:** the framework is complete and hardened (graph engine, control plane,
> data plane, sub-patches, undo/redo, both Python node tiers, e2e). The **node
> library is a deliberate blank slate** — Oscillator and Buffer only — and is being
> re-designed from scratch rather than ported.

## Why the rewrite

The Python version ran one OS process per node and moved data between them over
shared memory, with a Python manager in the middle of every control *and* data path.
The Rust version collapses that into a single process:

- **One process, tiered execution.** Native nodes run inline on the tick thread;
  Python nodes run either in-process on a free-threaded interpreter (no GIL) or, when
  they need the GIL, on a detached off-tick worker over shared memory — so a slow or
  hung node can't stall the graph.
- **One stream per output slot.** Viewers publish a *ViewSpec* (a constraint algebra,
  no payload); the server folds every viewer's needs into a single reduction that runs
  off the tick path. Ten viewers on one slot cost one stream, not ten.
- **A CRDT control plane.** The browser holds a read-only replica of the graph state.
  Every mutation is a command with an exact inverse, applied by the manager, which
  also owns undo/redo per client session.

## Running it

Requires a Rust toolchain (1.89+) and Node.js. Python is optional — build with
`--no-default-features` for a pure-native binary.

```bash
cargo run -p goofi-init   # once per clone: provisions the Python interpreters (needs `uv`)
cargo run                 # builds the SPA if needed, starts the server, prints the URL
```

`goofi-init` is a workspace crate, not a shell script, so that first line is the same command in
PowerShell, cmd, bash, zsh and fish. It is needed because pyo3 must be told which interpreter to
link against *before cargo starts*, and cargo reads `.cargo/config.toml` only at startup — until
it has run, the build stops with one line saying so.

Flags: `--port N` (default 8000), `--bind HOST` (default 127.0.0.1),
`--subproc-nodes DIR` / `--auto-nodes DIR`, `--subproc-python BIN`,
`--list-nodes`. With no `--*-nodes` flag it auto-discovers `./nodes/` and routes
each node to the tier it can run on. `--auto-nodes` is **repeatable** — each
directory is scanned in turn, and a later one wins a type name it shares with an
earlier one.

### In Docker

The image builds everything itself, so the host needs neither Rust nor uv nor Node:

```bash
docker build -t goofi .
docker run --rm -it -p 8000:8000 -v .:/workdir -v goofi-home:/home/goofi goofi
```

That second line is **literal** — no `$HOME`, no `$(id -u)`, no `~` — so it is the same text
in bash, zsh, fish, PowerShell and cmd. Docker resolves `.` itself, and a named volume is
keyed by name rather than by a host path. Any goofi flag appends: `docker run … goofi --port 9000`.

**`-v .:/workdir`** mounts the directory you launched from, and it is goofi's working
directory, so it appears in the Save/Load modal as *Working dir*. Patches saved there land
on your host.

**`-v goofi-home:/home/goofi`** is a Docker-managed volume holding the agent harnesses'
credentials. Log in to `claude`, `codex` or `opencode` once inside the terminal panel and it
persists — from any directory, because the volume is found by name. To use API keys from your
shell instead, opt in per variable (they are *not* passed automatically):

```bash
CLAUDE_CODE_OAUTH_TOKEN=… docker run … -e CLAUDE_CODE_OAUTH_TOKEN … goofi
```

To reach any other host directory, mount it at the same path on both sides — `-v /data:/data`
keeps a `.gfi` at `/data/x.gfi` meaning the same thing inside and out. Create the directory
first: Docker makes a missing mount source itself, owned by root, which a non-root container
then cannot write.

**Anywhere else, without a mount.** The Save and Open dialogs each carry a second door —
*Download a copy* and *Open from this computer…* — which pass the `.gfi` through the browser
rather than the backend. The browser runs on your host, so its own file dialogs reach any
location, mounted or not. This is a copy out and a copy in: it deliberately leaves the patch's
remembered file alone, so Ctrl+S never silently retargets to a download.

Notes: on macOS and Windows, run it from a WSL2 or POSIX shell — Docker Desktop maps mount
ownership itself, so `--user` is unnecessary there. If a host uses a uid other than 1000, add
`--user "$(id -u):$(id -g)"`. And `docker` never inherits your shell's environment: `-e NAME`
without a value passes that variable through, and is a silent no-op when it is unset.

### Python nodes

Drop a file in `nodes/`:

```python
import goofi
import numpy as np


class Smooth(goofi.Node):
    """Rolling mean over the last axis."""

    @staticmethod
    def config_input_slots():
        return {"data": goofi.DataType.ARRAY}

    @staticmethod
    def config_output_slots():
        return {"out": goofi.DataType.ARRAY}

    @staticmethod
    def config_params():
        return {"smoothing": {"window": goofi.IntParam(8, 1, 512)}}

    def process(self, data):
        w = self.params.smoothing.window
        kernel = np.ones(w, dtype=np.float32) / w
        out = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), -1, data.data)
        return {"out": (out, data.meta)}
```

`process` receives one keyword argument per **declared** input slot and returns
`{slot: value}` — a bare array, an `(array, meta)` pair, or a `goofi.Data`. Returning
`None` emits nothing that tick. Params arrive as `self.params.<group>.<name>`.

A slot with no data arrives as `None`, and handling that is the node's own call
(`if data is None: return None`). Declare it
`goofi.InputSlot(goofi.DataType.ARRAY, required=True)` instead and the node never ticks
without it — so it may be read unconditionally.

The same file runs on either Python tier — a discovery probe imports it in a real
interpreter and reports whether it is free-threading-safe. If it isn't, it routes to a
subprocess automatically and appears in the palette under the `subprocess` category. A node
whose dependencies are missing everywhere fails its probe and is listed as `unavailable` —
greyed out, naming the missing module — rather than silently vanishing. An exception inside
`process()` surfaces on the node's error channel instead of taking anything down.

The interpreters are `.gfivenv-ft` (free-threaded 3.14t, in-process) and `.gfivenv` (a GIL
Python, subprocess). `cargo run -p goofi-init` creates both and installs the `goofi` package
into them, using `uv` — which is therefore a hard requirement. `--subproc-python` overrides the
GIL one. Re-run `goofi-init` after a version bump; it is idempotent.

## Development

```bash
cargo test --workspace                    # backend
cargo test -p goofi-py --features embed   # in-process Python host
cd frontend && npm run check && npm run test
cd tests/e2e && npm run e2e               # Playwright against the real binary
```

### Layout

```
backend/
  goofi-core      Data (always f32) + Meta, params, reduction kernels, globals
  goofi-codec     the binary wire format (shared with the frontend decoder)
  goofi-node      the Node trait, manifests, Python introspection probe
  goofi-nodes     the native node library
  goofi-engine    Graph, scheduler, sub-patch scopes, commands + history, detached tier
  goofi-view      the ViewSpec constraint algebra
  goofi-crdt      the yrs document + reconcile
  goofi-bridge    axum server: control plane, data plane, SPA hosting
  goofi-py        in-process Python tier (pyo3, free-threaded)
  goofi-pymod     the `goofi` Python package, written in Rust
  goofi-subproc   the subprocess tier's manager side
  goofi-cli       the goofi-pipe binary
frontend/         SvelteKit SPA (the only UI)
tests/e2e/        Playwright end-to-end suite
nodes/            Python nodes, auto-discovered at startup
```

`AGENTS.md` is the working orientation for the repo and goes deeper on every
subsystem. (`CLAUDE.md` is a one-line `@AGENTS.md` import, so Claude Code finds
it too — the same pairing goofi seeds into a patch workspace.)

## License

MIT — see [LICENSE](LICENSE).
