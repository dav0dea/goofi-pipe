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
`--extra-nodes DIR`, `--list-nodes`. It scans `./nodes/` when that directory exists,
routing each node to the tier it can run on — in-process when its imports are
free-threading-safe, else a subprocess. Neither the tier nor the interpreter is a
setting: one probe decides per node, and the subprocess tier always runs `.gfivenv`.
`--extra-nodes` is **repeatable** and **adds** to the shipped tree: each directory is
scanned in turn, and a later one wins a type name it shares with an earlier one.

**When the backend is not on your machine.** The Save and Open dialogs each carry a second
door — *Download a copy* and *Open from this computer…* — which pass the `.gfi` through the
browser rather than the backend, so they reach wherever your own file dialogs do even when the
server cannot. This is a copy out and a copy in: it deliberately leaves the patch's remembered
file alone, so Ctrl+S never silently retargets to a download.

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
into them, using `uv` — which is therefore a hard requirement. Those two are the only
interpreters goofi uses; there is no flag for naming another. Re-run `goofi-init` after a
version bump; it is idempotent.

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
  goofi-bridge    axum server: control plane, data plane, SPA hosting, the yrs document
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
