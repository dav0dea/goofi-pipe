# The graph, with more than one engine

`Graph` today is two things in one type: the model a patch IS, and the signal engine's scheduler.
That was free while there was one engine. The audio engine makes the seam real, and one part of it
is a prerequisite rather than a tidy-up.

## The classification

**Graph management, and it stays where it is.** The one `IndexMap<Uid, NodeEntry>` and `Kind`;
`links`; `scope_of`; uid minting and naming; the type catalog (`dyn_types`, `unavailable`,
`patch_types`); `globals`; `arrangement` and `viewpoint`; the whole sub-patch machinery; slot faces;
link legality; `fragment` / `serialize` / `load_doc`; the `nd()` rename rewrite.

**Signal-engine scheduling, and a second engine needs its own of each.** `wire: WirePlanner`;
`instance` (the iceoryx2 name scope); `spawn_host` and `NodeHost`; the service-name derivation;
`drain_status` / `apply_status` / `wire_ack` / `advance_wire` / `replan*` / `desired_wires` /
`slots_touching`; the generation counter; and on `Leaf` — `stage`, `ufreq`, `setup_error`,
`last_error`, `error_since`, `evaluated`, `param_errors`.

About a dozen functions do both in one body, and they all have one shape: **change the model, then
tell the engine.** `create_node`, `remove_node`, `restart_node`, `add_link`, `remove_link`,
`update_param`, `set_expression`, `rename_node`, `clear`, `load_doc`, `refresh_param`,
`apply_status`. That regularity is the good news — the seam is one call at the end of each body.

## Locked decisions

**A settle point, and this one is a prerequisite.** There is none today: `add_link` re-plans inline
per link, `load_doc` calls `add_link` in a loop over every persisted link, and `remove_node` re-plans
per dropped link. For an async engine that is merely wasteful. For a synchronous one it is N
topological sorts and N plan publishes per load, every intermediate a graph nobody asked for — which
is exactly what AGENTS.md principle 8's second half forbids. It goes in `resync_and_broadcast`, which
already runs after every write op, already takes graph-then-doc, and already re-derives the whole
projection. Also call it from the 1 ms drain, so a re-plan after a node reports `Ready` lands.

Two cheap things keep it from being forgotten. Each engine remembers a `topology_version`, so
`settle` is a free no-op when nothing moved and one harness assertion catches a miss. And laziness:
a forgotten settle leaves the audio thread on a stale plan, which is a visible test failure rather
than silent corruption — already better than today, where a forgotten `replan_behind` is silent.

**The seam is a five-method trait, not twelve notifications** — twelve notifications IS
decisions-from-unsettled-state:

    insert(uid, manifest, build, params) -> Option<String>
    remove(uid)
    settle(&GraphView)
    drain(&mut dyn FnMut(Uid, Status)) -> usize
    shutdown()

Create, remove and restart stay explicit, because a birth mints a generation and is not derivable
from settled state. Link, unlink, param, expression, rename, clear and load all collapse into
`settle`.

**The rejected alternative, and why.** "Each engine owns its own node set and observes the graph"
is a mirror, and **that mirror already exists and already costs**: `WirePlanner` holds `sinks`,
`generations` and `planned` keyed by `Uid` beside `nodes`, which is exactly why `remove_node` must
remember `wire.forget`, `restart_node` must call `wire.detach` at precisely the right moment, and
`clear` must call `reset_channels`. Do not add a second one. Engines as enum variants inside `Graph`
is rejected too: it puts iceoryx2 and a device library in one crate with the model.

**`Status` does not split, it moves.** `Ack` exists only for the async wire handshake; every other
variant — `Stage`, `Fault`, `Ufreq`, `BindingErrors`, `ParamValues`, `RefreshOptions` — is the shared
node-health vocabulary the frontend draws for any node. Two health vocabularies for one health
projection is worse than one enum with a variant the audio engine never sends. Move it from
`goofi-engine/src/runtime/wire.rs` to `goofi-node`, beside `NodeStage` and `NodeFault`, so the
graph's health record and the signal runtime both see it without an engine dependency.

**The audio engine's nodes are `Kind::Leaf`, not a fourth variant.** `Kind` has ZERO exhaustive
matches — every site uses `_ =>` or `matches!` — so a fourth variant compiles with no error and is
silently classified "not a leaf" by `contains`, `node_type`, `node_stage`, `output_slots` and
`fragment`'s param block. That is the boundary-port failure AGENTS.md records three times over. While
in the area, replace those `_ =>` arms with explicit `Kind::Facade | Kind::Port(_)` so the next
variant is a compile error. Do not add a sealed accessor or a `Face` enum — an abstraction for one
product.

**Do NOT carve `goofi-graph` out yet.** The audio crate's constraint — depends on `goofi-core` and
nothing above it — is satisfied the moment it has its own `Cargo.toml`; the manifest IS the
enforcement, since you cannot use tokio without adding a line. Carving `Graph` out first moves ~4000
lines and every import path for zero behaviour change. The boundary is worth having later, when it
has something real to forbid: a future graph op quietly calling `replan_slot`.

**One fix that the settle point needs.** `replan` always calls `wire.begin` and dispatches phase 2
even when the desired set is unchanged and both diffs are empty, so a whole-graph settle would send a
redundant `InSlot` and burn an ack per consumer slot. One short-circuit, one line.

## What this deletes

- `pillar` / `pillars` / `pillar_default`. Five hardcoded `"signal"` literals on the wire and in every
  `.gfi`, derived from nothing, read by nothing in the frontend, pinned only by two tests. The engine
  a node belongs to is a property of its type; the archive records nothing.
- `ParamSpec::Trigger` and `Param::Trigger`. No node declares one, and the Python probe schema has no
  `Trigger` variant at all, so a Python node cannot either — `Param::Trigger` is unconstructible. The
  whole `fire_triggers` flag on `param_value_json` / `param_from_json` exists only for it, with a
  working `Trigger.svelte` at the far end and no producer.
- The dead `required`-slot branches in `Buffer`, `Filter` and `Psd`: all three open with
  `let Some(d) = inp.get("data") else { return Ok(()) }` while declaring `required: true`, and the
  runtime raises the fault before `process` is ever called.
- `_replaceSnapshot`'s `wholesale` parameter, which both call sites pass and the body never reads.

**Not `Isolation`.** It looks decorative because the engine never reads it, but the bridge renders it
in `inspect_node` and derives `info["tier"]` from it, and it is the only place a user sees which
Python tier a node runs on.

## Open questions

- **Where `settle` gets its `GraphView`.** The engine needs the topology without being handed the
  whole `Graph`, and the cheapest honest shape has not been designed.
- **`slots_touching` is O(N × (links + bindings)) and runs once per node reaching `Ready`.** Nobody
  has noticed at signal-engine node counts. A large patch will.
- **`apply_status` writes the persisted param record** on `RefreshOptions` — a status handler
  mutating `.gfi` state. Any second engine that reports refreshed options inherits it.
- **`restart_node` bumps the wire generation** purely because iceoryx2 service names embed it. An
  audio node has no service names, so the bump means nothing there — but the `/data` reducer
  re-derives its subscribe address from that generation every second, so an audio node's viewer has
  to learn about a restart some other way.
- **`Graph::contains` means "is a running leaf", `exists` means "is any node", `wirable` means "leaf
  or port".** Three near-synonyms with different truth sets, and `Command::precondition` picks a
  different one per variant. Worth one pass to name them for what they answer.
