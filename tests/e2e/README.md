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
```

`npm run e2e` = `cargo build -p goofi-cli` (via the workspace) then `playwright test`.
Playwright spawns `goofi-pipe` on port 8399 (override `GOOFI_E2E_PORT`), waits
for it, runs the specs, and tears it down. One worker, headless browser, deterministic.

## Layout
- `playwright.config.ts` — port, `webServer` spawn/teardown, iceoryx2 SHM hygiene, and the four
  projects: `default` (every spec except `touch-*`), `touch` (only `touch-*`, Pixel 7 portrait),
  and `touch-landscape` (863×360) + `tablet` (712×1138), which run **only**
  `tests/touch-reflow.spec.ts`. The narrow scope is the point: the coarse media query answers the
  same at every touch geometry, so only what FITS is worth re-running.
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
- **The spawned backend runs without the pyo3 expression evaluator** (`numpy` isn't on its
  `PYTHONPATH`), so `param-expression evaluator unavailable` is expected and harmless — none
  of these flows evaluate expressions. A future spec that needs the evaluator must set
  `PYO3_PYTHON` + a numpy-bearing `PYTHONPATH` on the `webServer` command.
- Reads through the façade are asynchronous w.r.t. the CRDT doc round-trip: wait on
  `waitForNode`/`waitForNoNode` or `expect.poll(...)` rather than reading once.
