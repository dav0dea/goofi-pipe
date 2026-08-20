# goofi-pipe

A real-time, node-based data-processing platform for biosignals. A user builds **patches** in a
browser node-graph: each node ingests, transforms or emits `Data`, and edges carry data between
output and input slots. It targets live, high-rate streams — kHz EEG, HD video — with many
simultaneous viewers.

This branch (`rust-rewrite`) is a ground-up Rust rewrite of the backend. The original Python
implementation is deleted from this branch; it lives on `main`, and for reference at
`../../goofi-pipe/`. The SvelteKit frontend carried over and is the only UI.

**This file is orientation, and nothing else.** It holds the design principles and the
architectural decisions — the things the code cannot tell you because they are choices rather
than facts. Everything else is in the code: read that. A map of the crates, the shape of a
manifest, the name of an op — all of it goes stale here and never goes stale there.

---

## Communication

All communication — internal thought, explanation, planning, sub-agent exchanges, and messages to
the user — is held in **ASD-STE100 Simplified Technical English** only.

---

## Design principles

These are ours. They are the standard this codebase is held to — not a general-purpose
methodology, and not an installed skill. Where an outside practice disagrees, this section wins.

1. **One programmatic interface.** Everything goofi can do, it does through one op vocabulary.
   `/control`, `/mcp`, a script and a test are TRANSPORTS over one entry point, never four
   surfaces with four sets of behaviour. A capability only the UI can reach is a defect, and a
   test that needs a door of its own is telling you the API is incomplete.

2. **The code is the source of truth.** A comment that restates the code duplicates it, and the
   duplicate goes stale. Write a comment only for a deliberate choice that reads as WRONG without
   it — an order that looks arbitrary and is load-bearing, a tolerance that exists because of a
   defect, a platform quirk. The history of how a decision was reached is not a comment. If the
   code needs prose to be understood, make the code simpler.

3. **Make the error impossible, don't handle it.** Prefer a type, a bounded domain, a shared
   schema or an unconstructible invalid state over a runtime guard. Keep genuine boundary errors:
   a Python node CAN raise, so propagate it — never panic.

4. **Delete before adding.** The cost of a feature is the lines it leaves behind. A rewrite that
   is smaller is the fix; a fix that is bigger needs a reason. Two code paths that should agree
   get unified at one source of truth rather than patched in both.

5. **One artifact.** `goofi-pipe` is a single binary with the frontend compiled in. It serves the
   app and never opens one — the URL is printed, and the opening is the user's. `--headless`
   withholds the app entirely and serves the API alone.

6. **One system, several representations.** Phone and desktop, agent and human, are the same
   machinery with different presentations — never a second track, never a per-device guard.

7. **Root cause before fix.** Trace it to its origin and fix it there. Three failed fixes means
   the architecture is wrong; stop and reconsider it rather than adding a fourth patch.

8. **One owner, and decisions from settled state.** Two halves of one rule, and the pair the rest of
   this file leans on hardest.

   Every conceptual thing has exactly ONE owner of its state. Everything else derives it on read, or
   is a strictly one-way projection of it. A mirror, a cache beside its source, a count kept in two
   layers, a value both stored and derivable — each is a thing every future change must remember to
   update in lockstep, and eventually one does not.

   And a decision is taken from SETTLED state: a batch of mutations yields at most one decision,
   taken after the batch. Sequencing work off each individual mutation makes every intermediate a
   command, and an intermediate is a state nobody intended and nothing asked for.

   What it cost: "how many viewers want this stream" lived in three places, and each acted the
   instant it was touched. A viewer that detached and re-attached inside one render tick passed
   through zero on all three, so a batch that ended exactly where it started still closed the
   socket, destroyed the backend's reducer and dropped the cached frame — under every OTHER viewer
   of that slot. The end state was correct throughout, which is why every test passed.

   Both halves are audited for on their own, and a finding stands **regardless of its
   justification** — a good reason is an attribute of the duplication, never an exemption from
   naming it. "No consequence observed" is a finding too: the cost of this pattern is paid later,
   by whoever writes the change that forgets one holder.

---

## How we work

In priority order. These override speed.

1. **Test the software in use, not its functions.** A test launches the system, commands a range
   of actions through the one programmatic interface, and asserts the state that results. That is
   the standard — not a unit test per function, and not test-first as a law. A suite of green
   functions that break when assembled is the failure this replaces.
   - Every Rust test lives in one crate, which is separate from the crates it judges, so it
     reaches only public API and is structurally incapable of pinning implementation detail.
   - The suite is a short list of NAMED SITUATIONS, one file each. A test earns its place by
     covering a way the system is used; prefer one scenario crossing four layers to four tests
     that each pin one. When a bug is fixed, **extend the scenario that would have caught it**
     rather than adding a test beside it.
   - Svelte component glue cannot mount in a unit runner. Verify it by typecheck and a Playwright
     scenario, and keep the testable logic in a module a scenario can drive.

2. **A fixture that cannot express the failure makes a passing test theatre.** The question is
   never "does this stand in for the real thing?" but **"can this reproduce the failure I am
   trying to prevent?"** — and the way to answer it is to run the broken variant against your
   fixture and watch it pass. This has cost real defects: a fake socket hard-coded to OPEN hid a
   message dropped on a connecting one; a single-node fixture hid a counter summing across
   streams; a load test into a *fresh* instance passed against code that renumbered every node.

3. **Structural edits over shallow hacks.** Prefer the change that makes the codebase correct by
   construction over the one that silences the symptom. A larger, well-reasoned refactor is
   welcome when it removes a class of bugs — refactor scope is not gated.

4. **Deep code analysis.** Before changing a subsystem, hold enough of it in context to reason
   about the change's blast radius. Trust documented internal contracts; verify the ones you are
   about to depend on.

5. **Minimum diff, maximum clarity.** Match the surrounding idiom, naming and comment density.
   Do not reformat code you are not changing. Rust is 4 spaces; the frontend is tabs and single
   quotes. There is no rustfmt.toml and no Prettier config — **never run Prettier**, hand-match
   the style instead.

6. **Zero warnings, and that includes clippy.** A task is not done at "finished". Build with
   `--all-targets` and clear what it prints — that flag is load-bearing, because a plain build
   never compiles the integration test targets and a warning there ships. `cargo clippy --workspace
   --all-targets` is clean as of 2026-08-19 and stays that way. Remove the dead field; never
   silence it with a `_` prefix or an `#[allow]`.

7. **Honest reporting.** If tests fail, say so with the output. If a step was skipped, say that.
   State what is verified plainly; never claim done what you have not run.

**Hardening a subsystem is an audit run to convergence, not a read-through.** Fan finders across
its dimensions in parallel, then adversarially verify every candidate before believing it — a
correctness finding must trace a real caller and check the upstream guards, not just the local
function, and a leanness finding must not be a speculative reshape, which is itself inflation.
Re-run after each fix round. Convergence looks like the confirmed count shrinking *and* shifting
from structural to trivial; stop there rather than manufacturing churn. Use the most capable
model for every finder and verifier — a weaker finder under-finds.

---

## Architectural decisions

The shape of the system, stated where it is a CHOICE. The mechanism is in the code.

**One process, one graph, one document.** The graph, the engine and the web server live in one
Rust process. A browser's replica is READ-ONLY: every mutation is a command sent as an RPC, the
manager applies it, and the delta is broadcast. **The document is the only graph projection** — a
snapshot that also carried nodes was tried, and the two drifted.

**The document is plain JSON, and a delta is a merge patch.** It was a CRDT, and nothing a CRDT is
for was in use: the replica never writes, undo is the manager's own command history, and there is
nothing to merge. What was left was one-way replication, with the library fighting it — the
broadcast gate could not use the state vector, because a delete does not advance one. Merge patch
spends `null` on "delete this key", so it is exact only while the document has no null leaf; a test
pins that, and if a null is ever needed the delta needs an explicit tombstone instead.

**Several devices edit one patch at once, and the manager is what serialises them.** Concurrency is
resolved where the graph lock already is: an op applies, the delta is computed, and it is broadcast
— all with the document lock still held, so two writers cannot interleave into out-of-order
versions. A patch names the version it applies TO, which lets a replica tell the two ways a version
can mismatch apart. A patch it ALREADY holds is stale, because a socket is subscribed before it is
snapshotted, so a peer's edit in that window arrives twice; skipping it is routine. A patch reaching
PAST it is a lost delta, and it is refused rather than merged onto the wrong base.

**Everything is a command with an exact inverse.** Undo/redo is manager-owned and filtered per
session, so two browser tabs undo their own work. Layout is a document root like any other and
rides the same machinery. **Tolerance belongs to replay, strictness to the fresh caller:** a command's
own execution is idempotent so a stale toggle converges instead of wedging a session's stack,
and the first-hand RPC path gates on a precondition instead.

**No layout inverse restores raw state.** Every one re-plans through the forward planners. Pinning
an entry back into the slot it held resurrects what the forward op promoted away, on top of
whatever a peer has since built there.

**There is no tick.** Every node owns one thread and schedules itself, waking for a control
message, a frame on an input, or its own rate cap elapsing. Frames travel node to node over
shared memory, never through the graph — so no node runs under the graph mutex and no user action
waits on a `process()`. A node is KNOWN when its add answers and ADDRESSABLE only once it reports
ready; pub/sub has no history, so anything said before that is queued or re-planned, never lost.

**One data stream per (node, slot), whatever the viewer count.** Viewers publish a payload-free
constraint algebra; the bridge folds every viewer's constraints against the real frame and
reduces ONCE, on its own subscription — so no number of viewers can slow a `process()` down.

**Both Python tiers run one contract.** In-process free-threaded and subprocess GIL-bound nodes
share one marshalling seam, so they cannot drift. Neither the tier nor the interpreter is
selectable: one probe per node file routes it, by whether its imports keep the GIL disabled.

**Exit is a real teardown.** Every node is stopped and waited for — to a CEILING, not a join,
because a wedged node must not wedge the exit. That wait is what releases shared memory; what a
crash leaves behind is reclaimed by the next start's sweep.

**A patch is an archive.** A `.gfi` is a zip holding the manifest beside the workspace tree it was
saved with. A load extracts into a FRESH mount, parses, and only then swaps: graph and workspace,
or neither. A load restores the uids the patch was saved with, because everything keyed by uid
that the load does not itself remap depends on it.

**Identity is structural.** One MCP server per goofi instance, many addresses: a spawned harness
is handed a URL goofi minted for it, so identity never travels through the agent and there is
nothing to spoof. The Origin/Host allowlist covers every route including the WebSocket upgrades —
a drive-by guard, not authentication; this app stays single-user and local by design.

**The frontend is a replica, and its styling has one source.** Every colour, spacing, type and
motion token lives in one `:root`; a component states its own layout, never another's. The
primitive library is a LEAF layer and must not import a store — doing so reshuffles the CSS chunk
graph and gave the app a first-paint flash.

---

## Hard constraints

- **Work happens on `rust-rewrite`. Leave `main` alone.** Never push or force-push without
  explicit authorization; branch before committing on a default branch.
- **The version lives in ONE place** — `[workspace.package] version`. Every crate inherits it and
  the Python wheel derives it. Bumping it also re-provisions the venvs.
- Commit in small, focused, readable steps at green checkpoints — never one mega-commit. Commit
  messages end with a `Co-Authored-By:` trailer naming the model that actually wrote them.
- **`docs/` is gitignored on this branch.** It holds `roadmap/` — one file per major unbuilt
  feature — and nothing else. Do not restore what is not there.
- No auth on the WS endpoints — single-user, local/trusted-LAN app.
- **Responsive and touch-capable across desktop, tablet and phone**, both orientations. Desktop is
  the reference, but no interaction may exist solely behind hover, right-click or a keyboard
  chord. Touch needs its own door in, gated on the one spelling a test enforces. Panel width is
  independent of viewport width, so `@container` is the default tool and `@media` is reserved for
  real device-class questions. One theme, done well.
- **The workspace/panel system and the cable-drag feel are frozen UX.** Restyle them, do not
  redesign them. There is no phone-only layout mode: a phone renders the same panel tree, and
  panel maximize is the small-screen mechanism.
- **Navigation must not dirty the patch, on either platform.** Entering a sub-patch is navigation;
  changing a viewer's type is a real view setting. Persistence and dirtiness are separate axes,
  and an unclassified write counts as authoring — so forgetting to classify costs a spurious dot,
  never a lost change.
- Do not reintroduce dearpygui, zmq, or a Python manager.
- The shared-memory transport and the binary wire format are load-bearing; changing either means
  changing the frontend's decoder in lockstep, and a golden pins it.

---

## Getting it running

Setup is ONE command, and it is not a shell script: `cargo run -p goofi-init`, once per clone,
with `uv` on PATH. It creates both interpreters, installs both wheels, and writes the gitignored
cargo config that points pyo3 at the free-threaded one. Until it has run, the build fails with a
single line telling you to run it.

The gates, once it is provisioned:

```bash
cargo test --workspace                      # must stay green, and warning-free
cargo test -p goofi-tests --features embed  # …plus the in-process Python tier, which LINKS libpython
cargo build --workspace --all-targets 2>&1 | grep -n '^warning'   # anchor the grep: a test's own
#   log line can contain "warning:" and read as a failing gate when it is not
cargo clippy --workspace --all-targets                             # …and this prints nothing
cd frontend && npm run check && npm run test   # svelte-check + tsc strict, then vitest
cd tests/e2e && npm run e2e                    # Playwright against the real binary
```
**TypeScript stays on 6.x.** 7 installs and `svelte-check` will run against it — with both versions
side by side and a `--tsgo` flag — and it checks **66 files instead of 754** and reports success.
A gate that silently covers a tenth of the app is worse than no gate. Re-try when svelte-check's
TS 7 support stops being experimental; it is one version string.

Two interpreters, both machine-local and gitignored, and the names are deliberate — a generic
`.venv` is claimed by editors and by `uv` itself, and a stale one is not inert:

- a free-threaded 3.14t, which the in-process host LINKS against and the discovery probe uses;
- a GIL python, which the subprocess tier always runs, because that tier exists precisely for
  packages that are not free-threading-safe.

**Never resolve the interpreter path.** The config names it RELATIVE and canonicalizes nothing: on
unix a venv's `python` is a symlink into the base install, which is exactly where the goofi wheel
is not — so canonicalizing hands pyo3 an interpreter that cannot import it, nothing errors, and
every Python node silently drops to the subprocess tier.

**Known gap:** provisioning reinstalls when the wheel is missing or broken, never when it is
merely stale. After changing the Python package, delete the venv — or the probe keeps running the
old wheel and a node using a new authoring feature silently disappears from the palette.

The cross-language tests find these interpreters themselves and **fail with an actionable message**
when none can import goofi. They never skip, and nothing in the suite is `#[ignore]`d.

`/dev/shm/iox2_*` is not a leak, and two reviews have now misread it as one. The count PEAKS
during a run and settles back, because every node releases its shared memory when it drops. Delete
those files by hand only to get a clean measurement, never as a fix. Relatedly: **a struct that
owns an iceoryx2 node beside its ports must declare the node LAST**, because Rust drops fields in
declaration order and a node dropped first cannot remove its own directory.

Never background CPU load with `(cmd &)` subshells when benchmarking — leaked processes outlive
the test and corrupt every later measurement.
