# The graph, with more than one engine

`Graph` today is two things in one type: the model a patch IS, and the signal engine's scheduler.
That was free while there was one engine. The audio and graphics engines make the seam real.
Designed with the user 2026-08-29, over four rounds against the code; this file is the result.

## The three parts

**The graph** (`goofi-graph`) is the op authority for the MODEL. Every op is a command with an
exact inverse, serialized under one lock, producing one delta and one undo history — so op parsing
and execution stay graph-side, shared across engines, once. The graph holds what the `.gfi`
persists: nodes, names, positions, links, params-as-record, bindings-as-authored, scopes, layout,
globals, viewpoint. It routes each op's PROPAGATION to the engine that owns the node, through one
trait. Its manifest forbids it iceoryx2, threads and tokio — the boundary is enforced by
dependencies, not discipline.

**An engine** (`goofi-signal`, later `goofi-audio`, `goofi-graphics`) is the authority for RUNTIME
state: the node instances, their health, their within-engine transport, and its own node library.
Within-engine communication is the engine's own affair — self-scheduled threads over iceoryx2 for
signal, a topologically ordered single thread with arena buffers for audio and graphics. The graph
never sees it.

**The transport** (`goofi-transport`) is cross-engine communication, and it is the same for every
engine, forever: always iceoryx2. It holds the endpoint machinery (publishers, subscribers,
doorbells, latest-wins mailboxes — extracted from today's `IoxTransport`) and the resolver: PURE
functions from `(instance, uid, slot, generation)` to a service name. A phone book, not a
switchboard — no state, no request/response, callable by any engine and by the viewer reducer.
The signal engine embeds an endpoint per node (its within-engine transport IS the cross-engine
transport); a scheduled engine embeds one set per engine and drives its boundary edges from its
own scheduler.

## The flow of an op

Graph parses and applies to the model, then the owning engine propagates to its node. `param edit`
writes the record identically for every engine; the propagation is a `SetParam` control message
for signal and a CLAP param event on the audio thread — each engine's own translation, behind the
trait.

## Locked decisions

**The seam is a small trait, not twelve notifications** — twelve notifications IS
decisions-from-unsettled-state:

    insert(uid, manifest, build, params) -> Option<String>
    remove(uid)
    settle(&GraphView)
    drain(&mut dyn FnMut(Uid, Status)) -> usize
    library() -> ...          // the engine's node classes, advertised on request
    shutdown()

Create, remove and restart stay explicit, because a birth mints a generation and is not derivable
from settled state. Link, unlink, param, expression, rename, clear and load all collapse into
`settle`. `drain` stays a PULL — the bridge's existing 1 ms sweep calls it; a single-threaded
engine queues statuses on its own thread and hands them over. Engines are registered at the
composition root: `Graph::new(engines)`, keyed by the engine tag on the manifest. Adding an engine
is one line there plus tagged manifests; nothing engine-specific enters the graph.

**Cross-engine wiring is derived names from settled state, never a protocol.** After a mutation
batch, every engine reads the same settled `GraphView` (the WHOLE graph — engines filter). The
producer's engine sees a cross-engine edge leaving `(uid, slot)` and ensures a publisher exists
under the derived name, ringing the consumers' derived doorbells after each publish; the
consumer's engine subscribes to that same name. iceoryx2's `open_or_create` is the rendezvous —
whichever side settles first waits. No message between engines, nothing to time out. This makes
cross-engine wiring eventually-convergent rather than ack-phased: a mis-ordered settle costs at
worst one missed wake on a continuous stream. The intra-signal three-phase planner survives
unchanged INSIDE `goofi-signal` — it is the async engine's private mechanism.

**The generation counter moves from `WirePlanner` to the graph**, because both endpoint engines
and the `/data` reducer must compute the same name. This also answers how an audio node's viewer
learns about a restart: the same way every viewer does, through the generation in the derived name.

**Node state splits by WRITER, and each half is single-writer.** The mirror across the thread (or
process) boundary is unavoidable; principle 8 permits exactly one shape for it — a strictly
one-way projection. `Leaf` today mixes both directions in one struct:

- The RECORD (`manifest`, `params`, `bindings`) stays on the graph's `Leaf`: op-written,
  graph-owned, what the `.gfi` persists.
- A `Health` struct (`stage`, `fault`, `ufreq`, `evaluated`, `param_errors`) with exactly one
  write path: the engine's `drain`. The known violation gets fixed here, not inherited:
  `apply_status` writes the persisted param record on `RefreshOptions` — the drain must update
  the projection plane only, with record writes on the op path.
- `host` leaves `Leaf` entirely, behind the signal engine, along with `wire`, `instance`,
  `bind_keys` and `spawn_host`.

**`goofi-node` splits into the shared vocabulary and the signal author contract.** Stays (shared
across engines): `NodeManifest` + slot decls (gaining the engine tag), the param vocabulary
(`ParamGroups`, `ParamKey`, `ParamDecl`, `ParamSpec`, `Params`), the expression vocabulary
(`ExprDecl`, `BindingId`, `Compiled`, `EvalCtx`, `ExprEvaluator`, the scanners), plus the seam:
the `Engine` trait, `GraphView`, and the health vocabulary (`Status`, `NodeStage`, `NodeFault`)
moving in from `runtime/wire.rs`. Moves to `goofi-signal`: the `Node` trait and factories
(`setup()`/`process()` is the SIGNAL author contract — a CLAP plugin never implements it),
`Inputs`/`Outputs`, `RunPolicy` and the common decls (`autotrigger`/`ufreq` are the
self-scheduling model), `Isolation`, and all of `discover.rs` (the Python probe is signal library
enumeration). Ripple: `goofi-nodes`, `goofi-python` and the pymod depend on `goofi-signal`, which
is honest — they are signal nodes.

**Each engine owns its library.** The signal engine runs the Python probe and the `inventory`
enumeration; audio enumerates CLAP plugins; each advertises through the trait. The graph keeps the
one merged view the palette reads (`dyn_types`, `unavailable` machinery moves behind this door).

**One stage vocabulary, one health projection.** All engines share
`creating/setup/ready/error(derived)`; a synchronous engine simply never emits some stages
(its `insert` can fail inline). Addressability is engine-internal: the graph stops tracking
known-vs-addressable, and each engine gates its own dispatch. `Status` does not split, it moves:
`Ack` exists only for the async wire handshake, but two health vocabularies for one health
projection is worse than one enum with a variant the audio engine never sends.

**A settle point, and it is the prerequisite.** There is none today: `add_link` re-plans inline
per link, `load_doc` calls `add_link` in a loop, `remove_node` re-plans per dropped link. For a
synchronous engine that is N topological sorts per load, every intermediate a graph nobody asked
for. It goes in `resync_and_broadcast`, which already runs after every write op, already takes
graph-then-doc, and already re-derives the whole projection. Also call it from the 1 ms drain, so
a re-plan after a node reports `Ready` lands. Each engine remembers a `topology_version`, so
`settle` is a free no-op when nothing moved. One fix it needs: `replan` always calls `wire.begin`
and dispatches phase 2 even when the desired set is unchanged — one short-circuit, one line.

**The crates are carved NOW** — this supersedes the earlier "not yet": the moment the trait
exists, the boundary has something real to forbid, and the manifest is the enforcement. The
directory layout groups by engine (package names stay flat — Rust has no hierarchical crate
names — the hierarchy lives on disk):

    backend/
      goofi-core, goofi-node, goofi-transport, goofi-graph,
      goofi-bridge, goofi-cli, goofi-client, goofi-codec, goofi-view, goofi-init, goofi-tests
      signal/
        goofi-signal, goofi-python, goofi-pymod, goofi-nodes

Later engines land as `backend/audio/`, `backend/graphics/`. Dependency direction: engines and
the graph both look down at `goofi-node` and `goofi-transport`; neither sees the other's
internals; the composition root sees everything.

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

## The order decided

1. Split `Leaf` by writer and fix the `RefreshOptions` record-write, in place.
2. Introduce the trait + settle point, still one crate, suite green.
3. Carve the crates and the directory hierarchy — then almost purely a file move.
4. The delete list, riding wherever it touches.

Each step a green checkpoint.

## What this deletes

- `pillar` / `pillars` / `pillar_default`. Five hardcoded `"signal"` literals on the wire and in
  every `.gfi`, derived from nothing, read by nothing in the frontend, pinned only by two tests.
  The engine a node belongs to is a property of its type; the archive records nothing.
- `ParamSpec::Trigger` and `Param::Trigger`. No node declares one, and the Python probe schema
  has no `Trigger` variant, so `Param::Trigger` is unconstructible. The whole `fire_triggers`
  flag on `param_value_json` / `param_from_json` exists only for it.
- The dead `required`-slot branches in `Buffer`, `Filter` and `Psd`: all three open with
  `let Some(d) = inp.get("data") else { return Ok(()) }` while declaring `required: true`, and
  the runtime raises the fault before `process` is ever called.
- `_replaceSnapshot`'s `wholesale` parameter, which both call sites pass and the body never reads.

**Not `Isolation`.** It looks decorative because the engine never reads it, but the bridge
renders it in `node state` and derives `info["tier"]` from it — the only place a user sees which
Python tier a node runs on. It moves to `goofi-signal`; it does not die.

## Open questions

- The exact `GraphView` shape — the cheapest honest struct carrying nodes, links, engine tags,
  generations and the resolver inputs, without handing an engine the whole `Graph`.
- The `library()` return shape, and how a rescan (bundle install, `--extra-nodes`) flows through
  it to the graph's merged view.
- `slots_touching` is O(N × (links + bindings)) and runs once per node reaching `Ready`. Nobody
  has noticed at signal-engine node counts. A large patch will.
- `Graph::contains` means "is a running leaf", `exists` means "is any node", `wirable` means
  "leaf or port". Three near-synonyms with different truth sets, and `Command::precondition`
  picks a different one per variant. Worth one pass to name them for what they answer.
