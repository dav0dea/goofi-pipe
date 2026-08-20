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

## Running it

Requires a Rust toolchain (1.89+), Node.js, and [`uv`](https://docs.astral.sh/uv/).

```bash
cargo run -p goofi-init   # once per clone: provisions the Python runtime
cargo run                 # builds the SPA if needed, starts the server, prints the URL
```

Python is part of goofi, not an add-on: nodes are written in it, and params are expressions
it evaluates. `goofi-init` builds the two interpreters that run them and writes the cargo
config pointing pyo3 at them — which must happen *before cargo starts*, because cargo reads
`.cargo/config.toml` only at startup. Until it has, the build stops with one line saying so.
It is a workspace crate rather than a shell script, so that first line is the same command
in PowerShell, cmd, bash, zsh and fish.

| Flag | Default | Effect |
| --- | --- | --- |
| `--port N` | `8000` | The port to serve on. |
| `--bind HOST` | `127.0.0.1` | The address to serve on. Anything beyond this machine warns: there is no auth, and `/term` is a real shell. |
| `--extra-nodes DIR` | — | Scan `DIR` for Python nodes *after* `./nodes/`. Repeatable; a later directory wins a type name it shares with an earlier one. |
| `--list-nodes` | — | Print the registered node types and exit. |
| `--headless` | — | Serve the API alone — `/control`, `/data`, `/term`, `/mcp`. The app's routes are never mounted. |

`./nodes/` is scanned whenever it exists; no flag turns it on or off.

**When the backend is not on your machine,** the Save and Open dialogs each carry a
second door — *Download a copy* and *Open from this computer…* — which pass the `.gfi`
through the browser rather than the backend. This is a copy out and a copy in: it leaves
the patch's remembered file alone, so Ctrl+S never silently retargets to a download.

## Python nodes

Drop a file in `nodes/`:

```python
import goofi
import numpy as np


class Smooth(goofi.Node):
    """Rolling mean over the last axis."""

    INPUTS = {"data": goofi.InputSlot(goofi.DataType.ARRAY, required=True)}
    OUTPUTS = {"out": goofi.DataType.ARRAY}
    PARAMS = {"smoothing": {"window": goofi.IntParam(8, 1, 512, doc="Samples in the mean.")}}

    def process(self, data):
        w = self.params.smoothing.window
        kernel = np.ones(w, dtype=np.float32) / w
        return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), -1, data.data)
```

A node declares itself in constants, read once by the import — not in hooks. Each may be
omitted:

| Constant | Shape |
| --- | --- |
| `INPUTS` | `{slot: DataType}`, or `{slot: InputSlot(dtype, required=…, trigger=…)}` for the per-slot options. |
| `OUTPUTS` | `{slot: DataType}`. |
| `PARAMS` | `{group: {name: IntParam / FloatParam / BoolParam / StringParam}}`, read as `self.params.<group>.<name>`. |
| `PRODUCER` | `True` for a node that paces itself rather than waiting for a frame. |

`process` receives one keyword argument per **declared** input slot — a `goofi.Data`, or
`None` when the slot holds no frame. A `required=True` slot never arrives empty, so it may
be read unconditionally. It returns `{slot: value}`, or a bare value when the node has
exactly one output slot; a value is a `goofi.Data`, an `(array, meta)` pair, or an
array-like. Returning `None` emits nothing. `setup()` runs once, after the params are
seeded.

The same file runs on either tier, and the file does not choose: a discovery probe imports
it in a real interpreter and routes it in-process when its imports keep the GIL disabled,
else to a subprocess — where it appears in the palette under the `subprocess` category. The
two interpreters are `.gfivenv-ft` (free-threaded 3.14t) and `.gfivenv` (a GIL Python), both
made by `goofi-init`, and goofi uses no others. Re-run it after a version bump; it is
idempotent.

Nothing fails silently. A node whose dependencies are missing everywhere is listed as
`unavailable`, greyed out and naming the missing module; an exception inside `process()`
surfaces on the node's error channel instead of taking anything down.

## Testing

```bash
cargo test --workspace                        # backend
cargo test -p goofi-tests --features embed    # …plus the in-process Python tier
cargo clippy --workspace --all-targets        # prints nothing
cd frontend && npm run check && npm run test  # svelte-check, then vitest
cd tests/e2e && npm run e2e                   # Playwright against the real binary
```

## License

MIT — see [LICENSE](LICENSE).
