# The graph, with more than one engine

`Graph` today is two things in one type: the model a patch IS, and the signal engine's scheduler.
That was free while there was one engine. The audio and graphics engines make the seam real.
Designed with the user 2026-08-29 over four rounds against the code, then hardened by a 23-agent
adversarial audit the same day: five finders, every structural candidate verified against the code
by a skeptic, thirteen confirmed findings folded in below, four refuted and kept out.

## The three parts

**The graph** (`goofi-graph`) is the op authority for the MODEL. Every op is a command with an
exact inverse, serialized under one lock, producing one delta and one undo history — so op parsing
and execution stay graph-side, shared across engines, once. The graph holds what the `.gfi`
persists: nodes, names, positions, links, params-as-record, bindings-as-authored, scopes, layout,
globals, viewpoint — plus two process-lifetime birth facts the resolver needs: the `instance`
scope and every uid's generation. It routes each op's PROPAGATION to the engine that owns the
node, through one trait. It depends on nothing above `goofi-node` — `goofi-core` and `goofi-node` alone, never
`goofi-transport` or an engine — so the boundary is enforced by its manifest, not discipline: the graph provably never computes a service
name or touches an endpoint; it only carries the resolver's inputs in `GraphView`.

**An engine** (`goofi-signal`, later `goofi-audio`, `goofi-graphics`) is the authority for RUNTIME
state: the node instances, their health reports, their within-engine transport, and its own node
library. Within-engine communication is the engine's own affair — self-scheduled threads over
iceoryx2 for signal, a topologically ordered single thread with arena buffers for audio and
graphics. The graph never sees it.

**The transport** (`goofi-transport`) is cross-engine communication, and it is one shared
mechanism for every engine: iceoryx2 names and rendezvous, always. It holds the endpoint machinery
(publishers, subscribers, doorbells, latest-wins mailboxes — extracted from today's
`IoxTransport`) and the resolver: PURE functions from `(instance, uid, slot, generation)` to a
service name and its service CONFIG (buffer depth included — `open_or_create` lets either side
create the service, so the whole config must derive identically on both sides). A phone book, not
a switchboard. An in-process engine pair MAY carry the payload over a ring keyed by that same
derived name — the name and rendezvous are the invariant, not the copy. The signal engine embeds
an endpoint per node (its within-engine transport IS the cross-engine transport); a scheduled
engine embeds its own set and drives its boundary edges from its own scheduler. Engines and the
bridge (whose `/data` reducer holds subscribers) depend on it; the graph never does.

## The flow of an op

Graph parses and applies to the model, then the owning engine propagates to its node. `param edit`
writes the record identically for every engine; the propagation is a `SetParam` control message
for signal and a param value handed to the audio thread — each engine's own translation, behind the
trait, keyed by the touched set the op path records.

## Locked decisions

**The seam is a small trait, not twelve notifications** — twelve notifications IS
decisions-from-unsettled-state:

    insert(uid, type, generation, params) -> Option<String>
    remove(uid)
    settle(&GraphView, &[Touched])
    drain(&mut dyn FnMut(Uid, Status)) -> usize
    request(uid, Request)
    library() -> ...          // the engine's node classes, advertised on request
    shutdown()

Three defaulted doors landed beside these: `reset_clock` (a clear moved the patch origin),
`set_evaluator` (shared with every engine that evaluates bindings on its own thread), and
`universal_decls` (the engine's own universal group as DECLARATIONS — `with_common` moved behind
`normalize_params` for values, and the palette's tooltips read declarations through this). Plus
`as_any_mut`: the composition root's reach to a concrete engine's own surface — the runtime type
registry stays signal-concrete rather than growing trait vocabulary for one engine.

Create and remove stay explicit, because a birth mints a generation and is not derivable from
settled state — and `insert` carries that graph-minted generation, plus type identity the engine
resolves against its OWN library; never a build closure, which is signal vocabulary no other
engine can name. There is NO restart method: a restart is graph-side record work (the param fold,
the orphan-link prune, the rebind) plus trait-level remove+insert with a fresh generation — and
remove purges the engine's pending request queue for that uid, restart included: a held request
addresses an INSTANCE, and the instance a rebirth makes never asked for it. `clear` is N
explicit removes plus ONE settle; `load` is clear plus N explicit inserts plus one settle — a
removal derived from absence-in-the-view would be the engine-observes-the-graph mirror this file
rejects. Rename's `nd()` source rewrite stays on the op path; only its re-resolution rides settle.
`drain` stays a PULL — the bridge's drain worker calls it through a Graph method, woken by the
report-side notify; a single-threaded engine queues statuses on its own thread and hands them
over. The signal engine receives `instance` at construction, and the evaluator through the
`set_evaluator` door; `clear()`'s clock
reset reaches every engine through the trait's `reset_clock`, a default no-op for an engine with
no patch time. Engines are registered at the composition root — `goofi-bridge`'s `fresh_graph`,
the ONE boot path the CLI and the test harness share: it constructs the signal engine (whose
construction carries the boot reclaim sweep) and registers it first on a bare `Graph::new()`;
the shipped nodes reach every engine through its scan of the folders the root's `build.rs`
prebuilt (2026-09-02: `goofi-nodes` and its inventory are deleted; a shipped Rust node is a
`.rs` file the engine loads from goofi's build cache, `node-sources.md`). A bare graph is a MODEL — it serializes, and runs
nothing. A type's engine is WHICH
library advertises it — no tag field exists anywhere; two libraries claiming one name resolve to
the first registered advertiser, signal first. Adding an engine is one line there plus its
library; nothing engine-specific enters the graph.

**Settle carries a touched set, because a settled view has no delta.** Link, unlink and topology
collapse into settle bare — the signal planner's `planned` map is already their diff base. Param
and expression edits do NOT reduce to a view diff: a `SetParam` is a write with effects (it can
unbind, wake a parked node, re-fire `on_param_changed`), so the engine must know WHICH key moved,
and an engine-side last-shipped copy of every record would be the third mirror this design
forbids. So the op path — the one writer — records `Touched` entries naming the `(uid, ParamKey)`
and `(uid, slot)` items the batch changed, including the bindings the batch INVALIDATED (a globals
edit or a rename re-resolves `Value` vars on bindings whose own keys the batch never touched; the
op path already enumerates exactly these sets in the invalidation walk). One settle per batch,
from settled state, with the batch's change list — a list written by the one op path is not a
second owner. The graph owns the settled-state version the no-op memo compares.

**The drain contract: `Ack` and `Ready` never cross the seam.** The engine consumes both
internally — the graph has no planner to route an ack to and no attach map for a ready. Only the
health projection crosses: Stage, Fault, Ufreq, RefreshOptions, BindingErrors, ParamValues. At the
type level this is one nested enum, not two flat ones (which would duplicate the six health
variants — the two-vocabularies cost this file rejects): the signal wire protocol becomes
`WireStatus { Ack { seq, ok }, Ready, Health(Status) }` inside `goofi-signal`, and the shared
`Status` in `goofi-node` keeps only the six health variants.

**Nothing polls to discover; a clock only paces.** Cross-engine delivery is latest-wins
everywhere — engines do not run in sync, so a queue between them lies about time; the one
exception stays the explicit in-order bridge node. A consumer draining at its OWN clock — a
scheduled engine's tick, the `/data` reducer's frame pace — is pacing, not polling. The status
plane's 1 ms sweep is a poll-to-discover and becomes event-woken: a node's report also rings a
graph-side door, so the drain wakes on arrival (the stage broadcast keeps its 500 ms pace — that
is pacing). The trait's `drain` stays a pull; only the caller's wake becomes a notification.

**The Ready re-plan is engine-internal, and the no-op memo has two gates.** On draining a `Ready`,
the signal engine forgets the planned base for every touched slot and re-dispatches from its own
stored desired sets — from an EMPTY base, because a `Wire` carries no generation, so a rebirth
changes no desired set and no diff can express it. Settle is a free no-op only when the
settled-state version is unchanged AND the engine has no pending drain-side work of its own (a
Ready it drained, a rebirth it performed). In-flight sequences freeze their composed service names
at `begin` — sound because a generation only moves through an explicit insert, which forces a new
settle whose `begin` cancels the in-flight sequence; the engine keeps no generation mirror.

**Cross-engine wiring is derived names from settled state, never a protocol.** After a mutation
batch, every engine reads the same settled `GraphView`. The producer's engine sees a cross-engine
edge leaving `(uid, slot)` and ensures a publisher exists under the derived name, ringing the
consumers' derived doorbells after each publish; the consumer's engine subscribes to that same
name. iceoryx2's `open_or_create` is the rendezvous — whichever side settles first waits. No
message between engines, nothing to time out. Cross-engine wiring is therefore
eventually-convergent rather than ack-phased: a mis-ordered settle costs at worst one missed wake
on a continuous stream, and a subscriber sees only frames published after it exists — which is
the signal plane's own deliberate no-replay rule, inherited, not worsened. The intra-signal
three-phase planner survives unchanged INSIDE `goofi-signal` — it is the async engine's private
mechanism.

**Boundary delivery follows the consumer's clock, and no engine-level door exists.** A scheduled
engine's CLOCK thread has no doorbells: before each tick it drains what its boundary holds, so
every tick runs against the freshest cross-engine state, and a producer facing that thread rings
nothing. Which thread consumes is the engine's own affair, and audio's answer (2026-09-02,
`audio-engine.md`) is a control-half thread — an expression is Python and cannot evaluate on the
audio clock — so the audio engine IS doorbell-driven: its modulation thread is woken like a signal
node is, and hands the audio thread an atomic. An async consumer (signal) keeps its per-node door: a scheduled producer engine publishes under the
derived name and rings the consumer node's OWN doorbell with the event id `GraphView` carries,
exactly as a signal producer would — no intermediate engine proxy channel, no re-publish through
one. The producer's ring decision is one engine-tag branch on the settled edge, and the per-node
naming scheme is the only one the resolver needs.

**A cross-engine edge is latest-wins by decree.** An in-order crossing — the sample-carrying
signal-to-audio path — is an explicit BRIDGE node owned by the scheduled engine, generalizing
graphics-engine.md's clock-crossing decision to every scheduled engine; its edge's service config
(subscriber buffer depth) is a pure function of the settled edge, emitted by the resolver beside
the name. Modulation needs no bridge: it crosses as `nd()` at control rate, by audio-engine.md's
own decision.

**The generation counter and `instance` live on the graph**, because both endpoint engines and the
`/data` reducer must compute the same names; both ride `GraphView` as resolver inputs. The
invariant the planner enforces today moves with the counter and is stated here so it cannot be
lost: generations are PROCESS-LIFETIME — bumped on every birth at a uid, surviving `clear()` and
`load_doc` (a reloaded uid is born at generation+1, or it re-opens its predecessor's stale service
names), and never entering the archive. This also answers how an audio node's viewer learns about
a restart: the same way every viewer does, through the generation in the derived name.

**Node state splits by WRITER, and each half is single-writer** (LANDED, step 1). The mirror
across the thread (or process) boundary is unavoidable; principle 8 permits exactly one shape for
it — a strictly one-way projection. `Leaf` carries the two planes apart:

- The RECORD (`manifest`, `params`, `bindings`) stays on the graph's `Leaf`: op-written,
  graph-owned, what the `.gfi` persists.
- `Health` (`stage`, `fault`, `error_since`, `ufreq`, `evaluated`, `param_errors`) lives on the
  graph BESIDE the record — the op path must read `evaluated` to resolve `nd().params`
  references — and birth is modelled as construction, not mutation: an insert, restart or load
  REPLACES the node's Health with a fresh one whose only non-default field is insert's inline
  boot error, and the engine's `drain` is the only MUTATOR of an existing Health thereafter. A
  fresh struct has no corpse numbers to clear, so the reborn-node inspector defect stays fixed by
  construction. The known violation died here: the `RefreshOptions` answer no longer writes the
  persisted param record — see the request door below. Accepted cost of birth-as-construction: a
  rebirth resets the error-onset clock, so a standing error — a surviving binding error included —
  reads young after a restart.
- `host` leaves `Leaf` entirely, behind the signal engine, along with `wire` and `spawn_host`.

**Imperative requests have one door: `request(uid, Request)`.** A refresh is a one-time
"re-enumerate now" that settled state cannot express — the same argument that keeps insert
explicit. `RefreshParam` is the first variant (goofi-signal implements it as today's held
`wire.send`); the audio engine's `refresh: true` device params are the known second user. The
ANSWER arrives through drain long after the RPC returned, and its home is the drain-written
plane: a refreshable `Str` param's live options are an OVERLAY beside Health, keyed by ParamKey —
`describe_node_params` and `node state` read the record overlaid with it (the DOCUMENT
deliberately has no options field; the state-update echo is how options reach a client), the
`refreshed` echo queue lives on the same plane, and the drain never writes the record.

**The binding machinery cuts at resolution.** `ExprBinding` holds both halves today, so the cut is
named: graph-side (op-written, projection-read) are the authored `source`/`enabled`/
`triggers_process`, plus `terms`, `bind_error` and the compile handle — name resolution and
compilation read only model state, and compiling at the RPC is what returns a real error to the
fresh caller. Engine-side, re-derived at settle and keyed by `(uid, ParamKey)`, are wire
resolution, `bind_id` and `bind_keys` — the planner's own `Copy` index, stripped from the
graph-side record. Event-id ALLOCATION stays on the graph's binding-edit op path (the 65..=128
free list reads only graph state), and the ids ride `GraphView` beside the bindings — a foreign
producer engine must read a consumer's doorbell ids to ring them. `EventId` itself lives in the
shared seam (`goofi-node`), NOT in `goofi-transport` as first planned: `BoundVar` and `GraphView`
carry it, and the transport already imports `Uid` from the seam — the reverse edge would cycle.

**`GraphView` presents port-resolved leaf-to-leaf edges**, computed once by the graph at the
settle point; a port with nothing behind it resolves to NO edge, which IS the open-port answer,
because a slot message carries the full desired set and absence empties it — never raw links,
or every engine re-implements the relay walk and the three copies drift; a cross-engine edge
cannot even be CLASSIFIED from a link that ends at an engine-less port. Its nodes carry the Leaf
record INCLUDING the derived binding state (rewritten source, resolved vars with event ids,
compiled id, trigger flag) — "bindings-as-authored" must not be read as source-strings-only —
plus each node's engine id, generations, `instance`, and per-slot event-id inputs (an input slot's id is
its manifest position). The patch clock origin is NOT a settle input: it rides the explicit
insert path, as today's spawn does.

**`goofi-node` splits into the shared vocabulary and the signal author contract.** First — and
LANDED in step 1 — two fields came OFF `NodeManifest`: `factory` and `isolation` are the signal
engine's business — each engine's library maps a type name to its own factory and tier, and
`library()` advertises the tier as plain data so the bridge keeps its `node state` display.
Without that strip the split could not compile: the shared manifest named `Box<dyn Node>` and
`IsolationCell`, both of which move up. The inventory unit is now `NodeClass`. Stays in
`goofi-node` (shared): the stripped manifest + slot decls (NO engine tag was added — which
library advertises a type is its engine), `Uid`, the param vocabulary (`ParamGroups`, `ParamKey`,
`ParamDecl`, `ParamSpec`, `Params`), the expression vocabulary (`ExprDecl`, `BindingId`,
`Compiled`, `EvalCtx`, `ExprEvaluator`, the scanners), plus the seam: the `Engine` trait,
`GraphView`, `Touched`, `Request`, and the six-variant health `Status` with
`NodeStage`/`NodeFault`. Moves to `goofi-signal`: the `Node` trait and factories
(`setup()`/`process()` is the SIGNAL author contract — an audio node never implements it),
`Inputs`/`Outputs`, `RunPolicy` and the common decls, `WireStatus`, and all of `discover.rs`.
`Isolation` and `IsolationCell` stay shared after all: the seam's `LibraryEntry` carries the live
tier cell, so the bridge's `node state` reads one door for every engine. Ripple: `goofi-python` depends on `goofi-signal` — which is
honest, it runs signal nodes; the pymod does not (it imports nothing from `goofi-node` and stays
under `signal/` on disk unchanged).

**Each engine owns its library, and the rescan seam moves with it.** The signal engine runs the
Python probe and loads the Rust artifacts `goofi-build` made; audio enumerates its shipped nodes, the authored `cdylib`s and the VST3 plugins; each advertises
through `library()`. The graph keeps the one merged view the palette reads, and normalizes a
caller's partial params against what the owning engine's library entry declares (`with_common`
moved behind that door — common params are signal scheduling semantics an audio node will not
have). The runtime type REGISTRY is the signal engine's own surface, reached by the scan through
`as_any_mut`; the graph keeps only the unavailable overlay and the provenance (`patch_types`),
and orders the restarts the diff names. The second scanning engine arrived (audio, 2026-09-02),
so `Engine::scan(dir)` lands on the trait: a node source ROOT — the shipped tree, a bundle, the
patch's `workspace/` — holds one folder per engine named `nodes_<engine id>` (`nodes_signal`,
`nodes_audio`, later `nodes_graphics`), the bridge's `rescan` hands each engine its own folder
under each root, and the stamp baseline and the diff stay on the bridge's rescan path. `library
get`'s Python source-file resolution (`discover::camel`) moves engine-side with the scan.

**One stage vocabulary, one health projection.** All engines share
`creating/setup/ready/error(derived)`; a synchronous engine simply never emits some stages (its
`insert` can fail inline — and that error lands in the fresh Health at construction, so the one
error path holds). Addressability is engine-internal: the graph stops tracking
known-vs-addressable, and each engine gates its own dispatch.

**A settle point, and it is the prerequisite** (LANDED, step 2 first half). Inline re-planning is
gone: ops record `Touched` and `Graph::settle` — a public method — delivers each item once, from
settled state. `resync_and_broadcast` calls it after every write op, and the drain worker
calls it after applying statuses, so a re-plan that needs settled state lands without waiting for
an edit. The In-slot short-circuit landed with it, scoped to the wire plane (param delivery rides
`Touched`, not the wire diff, so the short-circuit cannot silence it). Two rules the build
surfaced, both landed: the drain-side settle must NOT deliver while an op batch is open — the
batch marker lives on the graph (`hold_settle`/`release_settle`), because the drain is another
thread and a thread-local cannot guard it — and a `Touched` entry naming a node the batch also
removed is dropped at settle, never delivered: remove purged the planner, and settle must not
repopulate it.

**The crates are carved NOW** — this supersedes the earlier "not yet": the moment the trait
exists, the boundary has something real to forbid, and the manifest is the enforcement. The
directory layout groups by engine (package names stay flat — Rust has no hierarchical crate
names — the hierarchy lives on disk):

    backend/
      goofi-core, goofi-node, goofi-transport, goofi-graph,
      goofi-bridge, goofi-build, goofi-cli, goofi-client, goofi-codec, goofi-view, goofi-init, goofi-tests
      signal/
        goofi-signal, goofi-signal-sdk, goofi-python, goofi-pymod

`goofi-codec` stays its own crate, beside `goofi-transport` — the extracted machinery moves
bytes and rings doorbells, so the transport ended up codec-free; the engines and the bridge
encode, and each depends on the codec itself. Considered and rejected: merging the two. The codec is a FORMAT contract — pinned by a golden, mirrored by the
frontend's TS decoder, and carrying the subprocess tier's `Request`/`Response` protocol — where
the transport is a mechanism that changes with iceoryx2; and the pymod wheel proves the seam:
its `extension-module` build uses codec + iceoryx2 while its FT-host rlib build uses the codec
and deliberately stays iceoryx2-free, which one merged crate could honor only by growing the
feature split that IS the two crates.

Later engines land as `backend/audio/`, `backend/graphics/`. Dependency direction: the graph
looks down at `goofi-core` and `goofi-node` alone; engines look down at `goofi-node` and
`goofi-transport`; the bridge additionally reaches `goofi-transport` for its reducer's
subscribers; neither graph nor engine sees the other's internals. `goofi-bridge`'s `fresh_graph`
is the engine-composition root; `goofi-cli` composes what only a process start holds — the
evaluator, the scan seam — and sees everything.

**No rename rides along.** "Type", "catalog" and today's identifiers stand; uniformity is the
bar, not new vocabulary — where two words exist for one thing, unify to the incumbent.

**The rejected alternative, and why.** "Each engine owns its own node set and observes the graph"
is a mirror, and that mirror already exists and already costs: `WirePlanner` holds `sinks`,
`generations` and `planned` keyed by `Uid` beside `nodes`, which is exactly why `remove_node`
must remember `wire.forget` and `clear` must call `reset_channels`. The engine's map behind the
trait is not a mirror of the model — it is the engine's own runtime state, and `settle` hands it
topology explicitly. Engines as enum variants inside `Graph` is rejected too: it puts iceoryx2
and a device library in one crate with the model. A stateful cross-engine broker is rejected for
the resolver: derived names need no negotiation.

**The audio engine's nodes are `Kind::Leaf`, not a fourth variant.** `Kind` has ZERO exhaustive
matches — every site uses `_ =>` or `matches!` — so a fourth variant compiles silently classified
"not a leaf". While in the area, replace those `_ =>` arms with explicit
`Kind::Facade | Kind::Port(_)` so the next variant is a compile error. No sealed accessor, no
`Face` enum — an abstraction for one product.

**The sibling roadmaps are amended to match** (done in the same commit as this file): audio's
control half is the engine's own main-thread side behind this trait — not a `NodeRuntime`
thread — and its dependency floor becomes `goofi-core` + `goofi-node` + `goofi-transport`
(none of which carries iceoryx2 threads or tokio into the DSP path, so the block-callback
testability claim survives); graphics loses its two stale sentences (the "CRDT doc", and the
archive as the reason the wire name is fixed — the true anchor is the engine's registered id,
which library advertises the type, and the dtype vocabulary).

## The order decided

1. DONE — Split `Leaf` by writer (Health as construction-then-drain), fix the `RefreshOptions`
   record-write via the options overlay, and strip `factory`/`isolation` off `NodeManifest` —
   all in place.
2. DONE — Introduce the trait + the settle point with `Touched`, still one crate, suite green:
   the settle point, the `Touched` plane, the `WireStatus` nesting, the `Engine` trait with
   `GraphView`, and the signal engine extracted behind it.
3. DONE — Carve the crates and the directory hierarchy: `goofi-transport` (the shared plane),
   `goofi-signal` under `backend/signal/` beside the Python family, and `goofi-engine` renamed
   `goofi-graph`, whose manifest holds the boundary — nothing above `goofi-node`.
   `tests/transport.rs` re-pointed in the same commit.
4. DONE — The delete list: the five `pillar` literals went, the manifest gained
   `goofi: "<version>"` read before the version gate, `Param::Trigger` and `fire_triggers` died
   end to end (the frontend's trigger control kind and `Trigger` primitive with them), the three
   dead required-slot guards now surface instead of silently returning, and `_replaceSnapshot`
   lost its unread parameter.

Each step was a green checkpoint, and each was followed by a three-lens audit whose findings are
folded in above. The readiness claim is PINNED: `goofi-tests/tests/engines.rs` registers two
skeleton scheduled engines — audio- and graphics-shaped, each publishing static data at its own
fixed tick — and walks the whole seam through the one op surface: the merged palette, birth and
health, a viewer on a foreign slot, data crossing both directions, `nd()` modulation landing
latest-wins, a rebirth with a fresh generation, and teardown.

## What this deletes

- `pillar` / `pillars` / `pillar_default`. Five hardcoded `"signal"` literals on the wire and in
  every `.gfi`, derived from nothing, read by nothing in the frontend, pinned only by two tests.
  The engine a node belongs to is a property of its type; the archive records nothing. The
  coupled half of audio-engine.md's one format change rides along: delete `pillar_default`, add
  `goofi: "<version>"`, stay at manifest version 1, read `goofi:` before the version gate so a
  refusal can name the writer.
- `ParamSpec::Trigger` and `Param::Trigger`. No node declares one, and the Python probe schema
  has no `Trigger` variant, so `Param::Trigger` is unconstructible. The whole `fire_triggers`
  flag on `param_value_json` / `param_from_json` exists only for it.
- The dead `required`-slot branches in `Buffer`, `Filter` and `Psd`: all three open with
  `let Some(d) = inp.get("data") else { return Ok(()) }` while declaring `required: true`, and
  the runtime raises the fault before `process` is ever called.
- `_replaceSnapshot`'s `wholesale` parameter, which both call sites pass and the body never reads.

**Not `Isolation`.** It looks decorative because the engine never reads it, but the bridge
renders it in `node state` and derives `info["tier"]` from it — the only place a user sees which
Python tier a node runs on. It becomes the tier the signal engine's library entries advertise; it
does not die.

## Open questions

- The `library()` return shape — the entries carry manifest, factory, tier and defaults; the
  scan-diff report rides the same door; the exact types are the build's to price.
- `keys_touching` — `slots_touching` as built, and the signal engine's rather than the graph's —
  is O(N × (links + bindings)) and runs once per node reaching `Ready`. Nobody has noticed at
  signal-engine node counts. A large patch will.
- `Graph::contains` means "is a running leaf", `exists` means "is any node", `wirable` means
  "leaf or port". Three near-synonyms with different truth sets, and `Command::precondition`
  picks a different one per variant. Worth one pass to name them for what they answer.
