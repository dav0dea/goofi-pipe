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

2. **The code is the source of truth, and comments are scarce.** Code that needs prose to be
   understood is code to simplify, not to annotate. The default is NO comment.

   The philosophy in this file is strong and we follow it, so code that follows it explains
   itself. What earns a comment is a **deviation** — the place we had a good reason to stray —
   and the comment states that reason in brief. This is rare by construction: a file thick with
   comments is either badly written or quietly off-philosophy, and both are the real finding.

   The limits are hard, not aspirational:
   - **An inline comment is one line. Two is the ceiling, ever.** Needing more means the code is
     wrong, or the reason belongs in this file as a decision.
   - **A docstring briefly states a purpose**, plus a few words on a parameter only where the name
     cannot carry it. A parameter that needs explaining is usually a parameter to rename.
   - **No comment should need extensive reading.** A reader skims it and moves on.

   Not comments: restating the code, the history of how a decision was reached, an argument for
   the design, a changelog, or a defect's biography. If it matters beyond the line it sits on, it
   is an architectural decision and belongs in this file.

3. **Make the error impossible, don't handle it.** Prefer a type, a bounded domain, a shared
   schema or an unconstructible invalid state over a runtime guard. Keep genuine boundary errors:
   a Python node CAN raise, so propagate it — never panic.

4. **Delete before adding.** The cost of a feature is the lines it leaves behind. A rewrite that
   is smaller is the fix; a fix that is bigger needs a reason. Two code paths that should agree
   get unified at one source of truth rather than patched in both.

5. **One artifact.** `goofi` is a single binary with the frontend compiled in — the server, and
   the CLI client of a running server, in one file. It serves the app and never opens one — the
   URL is printed, and the opening is the user's. `--headless` withholds the app entirely and
   serves the API alone.

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
   - **A situation is a SESSION, not an assertion.** One boot, then an ordered walk that stacks
     actions and probes after each, named by `test.step` so a failure still says which stage. The
     states worth testing have a history — a rename after an edit, a reconnect after a drop — and a
     test that boots, does one thing and exits cannot reach any of them. Measured here: the setup
     was 76% of a Playwright test and the assertion it existed for was 24%.
   - Inside a session, an assertion the next step DEPENDS on is hard; an independent observation is
     `expect.soft`, so one wrong reading cannot hide the ten behaviours after it.

2. **What e2e is for: the seam, and the app holding together.** Everything goofi can do is
   reachable through the op vocabulary and is proved against the manager in `goofi-tests` — far
   cheaper and far sharper than driving a mouse to reach the same op. So a browser test earns its
   place only where a browser is the instrument:
   - **The socket seam.** The frontend's client and the manager's document must fit with no slack
     and no overlap: every op lands exactly once, two tabs converge, a tab that loses the socket
     rejoins on the manager's document instead of merging its stale one. Each half's own suite
     passes with the other half broken, which is why this cannot live in either.
   - **Structural integrity.** One complex scene, swept for what a restyle must never be able to
     break: a page that scrolls, text clipped away, a control cut off by a box that cannot scroll,
     a tap target under the app's own `--hit`. **Never a design value** — no pinned padding, no
     token colour, no measured box. Design freedom is the point; the net catches things falling
     apart, not things changing.
   - **Gestures.** A door that only a finger opens is proved only by a finger.

   Everything else is comfort. Hundreds of tests re-driving the op surface through a mouse, or
   pinning a value a design pass will move, find nothing and cost the freedom to restyle.

3. **A fixture that cannot express the failure makes a passing test theatre.** The question is
   never "does this stand in for the real thing?" but **"can this reproduce the failure I am
   trying to prevent?"** — and the way to answer it is to run the broken variant against your
   fixture and watch it pass. This has cost real defects: a fake socket hard-coded to OPEN hid a
   message dropped on a connecting one; a single-node fixture hid a counter summing across
   streams; a load test into a *fresh* instance passed against code that renumbered every node.

4. **Structural edits over shallow hacks.** Prefer the change that makes the codebase correct by
   construction over the one that silences the symptom. A larger, well-reasoned refactor is
   welcome when it removes a class of bugs — refactor scope is not gated.

5. **Deep code analysis.** Before changing a subsystem, hold enough of it in context to reason
   about the change's blast radius. Trust documented internal contracts; verify the ones you are
   about to depend on.

6. **Minimum diff, maximum clarity.** Match the surrounding idiom and naming — but NOT its comment
   density, which is the one thing never to copy from a neighbour: comments are scarce by principle
   2, and a thick file is a file to thin, not a bar to meet.
   Do not reformat code you are not changing. Rust is 4 spaces; the frontend is tabs and single
   quotes. There is no rustfmt.toml and no Prettier config — **never run Prettier**, hand-match
   the style instead.

7. **Zero warnings, and that includes clippy.** A task is not done at "finished". Build with
   `--all-targets` and clear what it prints — that flag is load-bearing, because a plain build
   never compiles the integration test targets and a warning there ships. `cargo clippy --workspace
   --all-targets` is clean as of 2026-08-20 and stays that way. Remove the dead field; never
   silence it with a `_` prefix or an `#[allow]`.

8. **Honest reporting.** If tests fail, say so with the output. If a step was skipped, say that.
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

**A failed frontend build is a failed build.** The bundle is compiled into the binary, so there is
no such thing as falling back to the previous one — it is an app that does not match the binary
around it, and on a fresh clone it does not exist. The build script fails instead, and a binary
that ended up with no app refuses to start rather than answering every route with nothing.

**Headless is ONE mode with three doors.** `--headless`, `GOOFI_HEADLESS` in the environment, and
`GOOFI_HEADLESS` set for the BUILD — which leaves the app out of the binary and stamps
`HEADLESS_BUILD`, so that binary is headless for life rather than needing the flag repeated at
every run. All three fold into one boolean before anything reads it, because a mode reachable three
ways must not be three conditions to keep in step. That stamp is also what separates an empty
bundle someone ASKED for from one that is a broken build: the first is the mode, the second is
refused.

**A virtual node is a node, and only the backend may know otherwise.** A boundary port and a
sub-patch facade are nodes: named in the one namespace `nd()` reads, moved, wired, viewed, copied,
deleted and inspected by the same ops, and carried in the document's ONE `nodes` map. The backend
keeps the thin distinctions their own nature forces — neither runs, so neither holds params, a
manifest or a lifecycle stage, and a port relays rather than produces, so a read resolves what is
behind it. **The frontend gets none.** A frontend branch on port-ness or scope-ness is a defect
unless it is purely about how the thing is DRAWN.

The test for "is this necessary" is the UNWIRED state: a port with nothing behind it must be in the
state an unconnected leaf is in — present, addressable, viewable, wireable, saved — just with no
data. It is never absent, never an error, never a closed socket. Deleting the node behind a port
leaves the port; a viewer opened before the wire stays open and starts drawing when the wire lands.
What this cost, three times over: a port was DELETED when its target was, its `/data` socket was
refused with a terminal close code that the client then made permanent, and `node state` answered
"no node" for the thing `node add` had just returned.

**A patch is an archive.** A `.gfi` is a zip holding the manifest beside the workspace tree it was
saved with. A load extracts into a FRESH mount, parses, and only then swaps: graph and workspace,
or neither. A load restores the uids the patch was saved with, because everything keyed by uid
that the load does not itself remap depends on it.

**Identity is structural.** A spawned agent's identity travels in its ENVIRONMENT, minted by
goofi at the spawn: `GOOFI_SESSION` names the server, `GOOFI_ACTOR` names its own undo stack, and
the running binary's OWN DIRECTORY leads PATH, so `goofi` resolves to this very binary — nothing is
detected, nothing is templated, and there is nothing to spoof. It is the directory rather than a
launcher laid beside it, because a launcher is a script and a script has a dialect: a `goofi.cmd`
is what `cmd` reads and what no bash-family shell will, so the agent that goofi most exists to
serve found no `goofi` at all. Copying the binary somewhere neutral is not the way out either —
Windows loads a process's DLLs from the directory it runs out of. What an agent can reach, every local process
can already reach: `/exec` and `/mcp` extend the same trust `/control` always has, and a stopped
agent's environment naming the server until the kill lands is that same trust, accepted. The
Origin/Host allowlist covers every route including the WebSocket upgrades — a drive-by guard, not
authentication; this app stays single-user and local by design.

**The frontend is a replica, and its styling has one source.** Every colour, spacing, type and
motion token lives in one `:root`; a component states its own layout, never another's. The
primitive library is a LEAF layer and must not import a store — doing so reshuffles the CSS chunk
graph and gave the app a first-paint flash.

**The panel system is a dependency, not a subsystem.** Tabs, splits, maximize and the drag-and-drop
are `panelty` — an npm package with a repo of its own (`dav0dea/panelty`), which goofi consumes like
any other dependency. It holds NO tree: it raises an intent, and goofi's `LayoutHost` turns each one
into a single manager op, so the document stays the only owner of the layout. Its styling is a
CONTRACT rather than a shared `:root` — the package ships a `--panelty-*-default` for every token
and reads `var(--panelty-x, var(--panelty-x-default))`, and goofi maps its own tokens onto that in
one block, which a test pins in both directions. Work that belongs to the panel system is a release
of the package, never a patch in this tree.

---

## Hard constraints

- **Work happens on `rust-rewrite`. Leave `main` alone.** Never push or force-push without
  explicit authorization; branch before committing on a default branch.
- **The version lives in ONE place** — `[workspace.package] version`. Every crate inherits it and
  the Python wheel derives it. Bumping it also re-provisions the venvs.
- Commit in small, focused, readable steps at green checkpoints — never one mega-commit. Commit
  messages end with a `Co-Authored-By:` trailer naming the model that actually wrote them.
- **`roadmap/` is the backlog, and it is committed.** One file per unbuilt feature, in the repo
  root. It is the ONE place a deferred item is recorded — an item tracked anywhere else is an item
  that drifts. A file states the decisions already taken and what is open, never a plan.
- No auth on the WS endpoints — single-user, local/trusted-LAN app.
- **Responsive and touch-capable across desktop, tablet and phone**, both orientations. Desktop is
  the reference, but no interaction may exist solely behind hover, right-click or a keyboard
  chord. Touch needs its own door in, gated on the one spelling a test enforces. Panel width is
  independent of viewport width, so `@container` is the default tool and `@media` is reserved for
  real device-class questions. One theme, done well.
- `/dev/*` is development surface: `--debug` (or `GOOFI_DEBUG=1`) opens it, nothing else does.
  `/dev/ui` renders one sample of every `$lib/ui` export and is a real tool for UI work; the guard
  against a primitive nobody uses is a vitest check that the barrel has no orphan export.
- **The workspace/panel system and the cable-drag feel are frozen UX.** Restyle them, do not
  redesign them — and the panel system is `panelty`, so restyling it means the token contract, not
  its source. There is no phone-only layout mode: a phone renders the same panel tree, and panel
  maximize is the small-screen mechanism.
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
with `uv` and `npm` on PATH. It creates both interpreters, installs both wheels, installs the
frontend's dependencies, and writes the gitignored cargo config that points pyo3 at the
free-threaded one. Until it has run, the build fails with a single line telling you to run it.

**Two commands is the ceiling** — that one and `cargo run`. Every precondition cargo has and cannot
provide for itself belongs inside goofi-init, never in a third line of a README: a build script
that stops to name a second setup step is a build that did not have to stop. The frontend's
`node_modules` was that third line, and it cost a fresh clone a server that started and served
nothing.

**The toolchain is pinned**, in `rust-toolchain.toml` — a different statement from `rust-version` in
Cargo.toml, which is the OLDEST compiler this code supports where the pin is the one it is built and
gated with. It exists because CI's `stable` had drifted eight releases past the machine the code was
checked on, so every lint added in between arrived as a CI failure on code nobody had touched.
Moving the pin is a deliberate commit that fixes whatever the new release names.

The gates, once it is provisioned:

```bash
cargo test --workspace                      # must stay green, and warning-free
cargo test -p goofi-tests --features embed  # …plus the in-process Python tier, which LINKS libpython
cargo build --workspace --all-targets 2>&1 | grep -n '^warning'   # anchor the grep: a test's own
#   log line can contain "warning:" and read as a failing gate when it is not
cargo clippy --workspace --all-targets                             # …and this prints nothing
cd frontend && npm run check && npm run test   # svelte-check + tsc strict, then vitest
cd tests/e2e && npm install && npm run e2e     # Playwright: its own package, its own install
#   Four situations across four viewport projects: the socket seam, structural integrity, gestures,
#   and the agent harness. Everything else, the op vocabulary already proves in goofi-tests.
```
**CI runs this list and nothing else** — `.github/workflows/ci.yml`, ONE job, because the gates
share one machine's worth of setup and the SPA is compiled in: no cargo build here happens without
the frontend's dependencies. It is the same list because a gate with two spellings drifts, and the
one that drifts is the one nobody runs by hand. The clippy line is spelled `-- -D warnings` there:
"and this prints nothing" is not enforceable by reading, and clippy carries the rustc lints too, so
that one command is the build-warning gate as well.
**That one job runs on Linux, Windows AND macOS**, as a matrix rather than three jobs, so there is
still one list — minus two steps that provably cannot differ. `svelte-check` type-checks, and a type has
no platform; Playwright drives a browser, and the half of it that IS platform-specific is the
binary underneath, which `goofi-tests` already proves on every runner. **vitest is NOT in that
set** and runs everywhere: its guards WALK THE TREE, and a `rel` built with `\` is how three of
them failed on Windows while one of those quietly found nothing and passed.
The matrix costs nothing — standard runners are free on a public repo — and each machine buys
something different. Windows is a whole separate PAL, and none of it is visible from Linux: a
ConPTY that answers a cursor query and never reports EOF, a `cmd` launcher, `\` in every path a
test compares, a `.pdb` two targets can collide on. Every one of those was found by hand, on a
machine, after it had already shipped. macOS is there for its ARCHITECTURE rather than its unix:
`macos-latest` is arm64, so it is the only place aarch64 is compiled at all, and the only place
pyo3's embedded interpreter meets dylib and codesigning rules. Note what the warning gate still cannot hold: `-D warnings` is
rustc's, and a CARGO warning — the `.pdb` collision was one — passes it silently.
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
those files by hand only to get a clean measurement, never as a fix — and NEVER from a script:
unlinking a same-user tmpfs file always succeeds, mapped or not, so no sweep can tell a corpse from
a live sibling's segment. The e2e harness had one, and every suite run severed each OTHER goofi on
the machine: its existing wires kept flowing on their mappings while every wire made after landed
on recreated, empty backing — no error, node green, data dead. Reclaim goes through iceoryx2
(`reclaim_stale_resources`, at every boot), which knows dead from alive; nothing else deletes. Relatedly: **a struct that
owns an iceoryx2 node beside its ports must declare the node LAST**, because Rust drops fields in
declaration order and a node dropped first cannot remove its own directory.

Never background CPU load with `(cmd &)` subshells when benchmarking — leaked processes outlive
the test and corrupt every later measurement.
