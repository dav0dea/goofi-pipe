# goofi e2e (Playwright)

End-to-end tests that boot the real `goofi-pipe` backend and drive the real SPA through
`window.goofi` (the agent façade). Committed, CI-runnable, isolated from `crates/` and
`frontend/`.

## Run

```bash
cd tests/e2e
npm install                      # first time
npx playwright install chromium  # first time
npm run e2e                      # builds the backend, then runs the suite
npm run gallery                  # just the two backend-free gallery projects (~8s inner loop)
```

`npm run e2e` = `cargo build -p goofi-cli` (via the workspace) then `playwright test`.

**One backend per worker.** `globalSetup.ts` spawns a fleet of `goofi-pipe`s on
`8500 + <worker slot>` (override the base with `GOOFI_E2E_PORT`, the width with
`GOOFI_E2E_WORKERS`) and reaps them afterwards; each worker derives its own port from
`TEST_PARALLEL_INDEX`, so no spec knows a port exists. That is what makes the suite parallel: a
worker still owns its backend alone, so the hermeticity rules below are unchanged — what a spec
leaks reaches the other specs on *its* worker. `--workers=1` still works and lands on the base port.
Each backend's stdout goes to `test-results/backend/backend-<slot>.log` (it no longer interleaves
into the report), and the stale-`iox2` sweep runs once in `globalSetup`, before the first spawn.

## Layout
- `playwright.config.ts` — the per-worker port arithmetic, the worker count, and the six
  projects: `default` (every spec except `touch-*` and the galleries), `touch` (only `touch-*`,
  Pixel 7 portrait), `gallery` + `gallery-touch` (the four `/dev/*` gallery specs, the only
  `fullyParallel` projects — they own no backend state), and `touch-landscape` (863×360) +
  `tablet` (712×1138), which run **only** `tests/touch-reflow.spec.ts`. The narrow scope is the
  point: the coarse media query answers the same at every touch geometry, so only what FITS is
  worth re-running.
- `globalSetup.ts` — the backend fleet: spawn, readiness, per-worker logs, iceoryx2 SHM hygiene,
  and the reaper Playwright runs as global teardown.
- `lib/app.ts` — `waitForApp` readiness gate (catalog arrived over the control WS).
- `lib/goofi.ts` — typed thin wrappers over the `window.goofi` façade, plus `waitForNode` /
  `waitForNoNode` (the doc round-trip must land before a just-added/removed node is read).
- `lib/touch.ts` — real CDP touch input (`page.mouse` still reports `pointerType: 'mouse'`, which
  is exactly what the coarse doors stand down for) plus `emptySpot`.
- `lib/topbar.ts` — the app header's action priority order and its overflow-menu row labels, shared
  by the three specs that ask the same question of the bar at different widths.
- `lib/uiSweep.ts` — the shared primitive-sweep fixture (`SAMPLES` / `exportedPrimitives` /
  `controlLocator`) driven by both gallery specs. Its source of truth is `$lib/ui/index.ts`: a
  primitive added to the barrel without a sample fails the sweep instead of escaping coverage.
- `tests/*.spec.ts` — the control-plane flows (boot, graph, undo, globals), the panel/chrome specs,
  the ui + inspector galleries, and the `touch-*` set.

## Notes
- **The spawned backends run without the pyo3 expression evaluator** (`numpy` isn't on their
  `PYTHONPATH`), so `param-expression evaluator unavailable` is expected and harmless — none
  of these flows evaluate expressions. A future spec that needs the evaluator must set
  `PYO3_PYTHON` + a numpy-bearing `PYTHONPATH` on the fleet's `spawn` in `globalSetup.ts`.
- Reads through the façade are asynchronous w.r.t. the CRDT doc round-trip: wait on
  `waitForNode`/`waitForNoNode` or `expect.poll(...)` rather than reading once.
