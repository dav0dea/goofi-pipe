# goofi e2e (Playwright)

End-to-end tests that boot the real `goofi-pipe` backend and drive the real SPA through
`window.goofi` (the agent façade). Committed, CI-runnable, isolated from `backend/` and
`frontend/`.

## Run

```bash
cd tests/e2e
npm install                      # first time
npx playwright install chromium  # first time
npm run e2e                      # builds the frontend + backend, then runs the suite
npm test                         # just `playwright test`, against whatever is already built
```

`npm run e2e` = the frontend build plus `cargo build -p goofi-cli`, then `playwright test`.

**One backend per worker.** `globalSetup.ts` spawns a fleet of `goofi-pipe`s on
`8500 + <worker slot>` (override the base with `GOOFI_E2E_PORT`, the width with
`GOOFI_E2E_WORKERS`) and reaps them afterwards; each worker derives its own port from
`TEST_PARALLEL_INDEX`, so no spec knows a port exists. That is what makes the suite parallel: a
worker still owns its backend alone, so the hermeticity rules below are unchanged — what a spec
leaks reaches the other specs on *its* worker. `--workers=1` still works and lands on the base port.
Each backend's stdout goes to `test-results/backend/backend-<slot>.log` (it no longer interleaves
into the report), and the stale-`iox2` sweep runs once in `globalSetup`, before the first spawn.

## Layout
- `playwright.config.ts` — the per-worker port arithmetic, the worker count, and the four
  projects, which are the GEOMETRIES the four situations are asked in: `desktop` (every spec except
  `touch`), `phone` (`touch` + `integrity`, Pixel 7 portrait), and `phone-landscape` + `tablet`,
  which re-ask `integrity` alone. The narrow scope is the point: `integrity` is the responsive
  test, and re-asking it is what makes it one.
- `globalSetup.ts` — the backend fleet: spawn, readiness, per-worker logs, iceoryx2 SHM hygiene,
  and the reaper Playwright runs as global teardown.
- `lib/app.ts` — `waitForApp` readiness gate (catalog arrived over the control WS), plus the
  split/tab helpers the shell specs share.
- `lib/goofi.ts` — typed thin wrappers over the `window.goofi` façade, plus `waitForNode` /
  `waitForNoNode` (the doc round-trip must land before a just-added/removed node is read).
- `lib/raw.ts` — a `/control` socket of the test's OWN, which is what lets the seam spec ask the
  manager what it holds instead of asking the accused to testify.
- `lib/touch.ts` — real CDP touch input (`page.mouse` still reports `pointerType: 'mouse'`, which
  is exactly what the coarse doors stand down for) plus `emptySpot`.
- `lib/invariants.ts` / `lib/geometry.ts` / `lib/inspector.ts` — the structural sweep: box
  arithmetic and the checks for a page that scrolls, text clipped away, or a tap target under
  `--hit`. Never a design value — the net catches things falling apart, not things changing.
- `lib/harness.ts` — the agent-harness helpers `agent.spec.ts` drives.
- `tests/*.spec.ts` — the four situations: `socket` (the seam), `integrity` (structural), `touch`
  (gestures), `agent` (the harness). Everything else, the op vocabulary already proves in
  `goofi-tests`.

## Notes
- **The spawned backends run without the pyo3 expression evaluator** (`numpy` isn't on their
  `PYTHONPATH`), so `param-expression evaluator unavailable` is expected and harmless — none
  of these flows evaluate expressions. A future spec that needs the evaluator must set
  `PYO3_PYTHON` + a numpy-bearing `PYTHONPATH` on the fleet's `spawn` in `globalSetup.ts`.
- Reads through the façade are asynchronous w.r.t. the document round-trip: wait on
  `waitForNode`/`waitForNoNode` or `expect.poll(...)` rather than reading once.
