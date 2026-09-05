# Param sources: constant, expression, reference

Every param of every node, on every engine, has exactly ONE active source. Decided with the user
on 2026-09-02 while designing the audio engine, because it is what gives that engine control-rate
and audio-rate modulation through one door with no precedence rule between them — and it applies
to the whole system, not to audio. BUILT 2026-09-02, as the first step of the audio program; what
is below is what landed.

## Decisions

**A param holds four things: its literal, its expression, its reference, and a mode.** The mode
is one of `constant`, `expression`, `reference`, and it names the active source. The other two
KEEP their content, so switching among the three is never destructive — the promise the fx toggle
already makes for the expression text, extended to the reference. `enabled` dies; the mode
replaces it.

- **constant** is the literal: what every param is at birth, and what the node reads whenever the
  active source cannot resolve.
- **expression** is Python over `nd()`, `globals` and `me`, evaluated on arrival at control rate —
  today's binding, unchanged. Every expression is ONE line; the expanded multi-line editor is
  deleted with the editor's `multiline` option.
- **reference** is ONE producer output slot and no Python. The param takes the producer's value at
  the producer's rate. A reference is what a cable is, for a param.

**A pulse is a request, never a value** (landed 2026-09-05). `ParamSpec::Pulse` holds no value:
its record carries only a source (`mode`, `expr`, `ref`, `triggers`), so a reference on it
survives a save and a `.gfi` never carries a fired state. The op `node param pulse` fires it
once, through the same request door as a refresh, and it changes no document state, so it is
not a command and leaves no undo entry. A reference or an expression on a pulse fires on a
rising edge — zero or false to non-zero or true — detected at the param boundary, once, for
every engine. A Rust node receives `on_pulse`; a Python node declares `goofi.PulseParam(doc=...)`
and defines `pulse_<group>_<name>(self)` beside `refresh_<group>_<name>(self)`; an audio node
gets it through its control half; the inspector draws a button.

**A reference is the string `node.slot`, by name**, in the record, the document, the `.gfi` and
the op. A rename rewrites it exactly as it rewrites `nd('name')` in an expression — one rename
mechanism. A deleted producer leaves the reference in place with a binding error and the literal
stands, as an unresolved `nd()` does; undo restores it. Nothing is pruned.

**Node names and slot names are `[A-Za-z][A-Za-z0-9]*` and not a Python keyword.** An expression
reads a slot as an attribute, and the rule is what makes `node.slot` unambiguous with no quoting.
One function in `goofi-core`, beside `is_valid_identifier` (which `globals` keep, underscore
included), enforced at `node add` and at rename — refused, never silently swapped for a minted
name — by the Python probe (a bad slot name registers the type UNAVAILABLE with the name quoted),
and by a test over every shipped manifest. A name the graph MINTS passes the rule by construction,
so a `_`-prefixed type is born `testscalar0`, never `_testscalar0`. `in` is a keyword and therefore
not a slot name. Not yet held for every engine: the graph checks nothing where it takes an engine's
library, so a Rust manifest is held only by the test — the dynamic-node step, where every Rust node
enters through one scan door, is where that door enforces it.

**Typing is by slot kind.** `Float`, `Int` and `Bool` reference an `Array` or an `Audio` output;
`Str` references a `String` output. A mismatch is a bind error at the op, and the literal stands.
On the signal plane a referenced `Array` frame must hold ONE element; a wider frame reports a
shape error through `BindingErrors` on arrival, the literal stands, and an expression with a
reduction is the tool.

**`triggers` applies to expression and reference alike.**

**The op is `node param edit`, grown by one flag**: `reference: string` beside `expression`, and
`mode` accepts `reference`. Giving a reference implies the mode, as giving an expression does;
an empty reference clears it; a mode alone switches among what is retained.

**The document shape is `{value, mode, expr, ref, triggers}`** — `mode` present when a record
exists, `expr` and `ref` as strings when they have content, `triggers` when true. The `.gfi` and a
copied fragment carry the same record as `sources: [{group, name, mode, expression, reference,
triggers}]`; a paste rewrites the names a source spells to the copy's, and a slot label only where
it IS a copied port's name, on the copied facade that holds it. The descriptor's fields are `mode`,
`expression`, `reference`, `triggers`, `error`. No shim for the old `enabled` shape: the pre-launch
policy applied. The record is ONE Rust type, `SourceState`, which the graph's record, the command
and the projection all carry rather than restate.

**A reference reaches the seam as one `BoundVar::Stream` with no compiled id**, and that absence
IS the kind — the seam ships no second field for it. The signal engine subscribes a reference like
a binding var and COPIES the scalar on arrival — no evaluator, no GIL, none of the `codes` mutex.
Re-pointing a reference at another producer starts it EMPTY: a held value carries across a rebind
only from the same stream, so a silent producer never stands in for the one it replaced.
The audio engine makes a same-engine reference a plan edge (per sample, per channel: a 4-channel
gate referenced into an envelope is four voices) and lands a foreign reference or an expression as
an atomic at control rate.

**The inspector's fx badge became three chips**: constant, expression, reference. Reference mode
is two fields side by side, node then slot, each the ONE expression editor component in its picker
configuration: no language, no highlighting, one completion source over the catalogue the
expression editor already reads, filtered by what the param may reference. A pick commits; a
producer with one matching output fills the slot itself. A reference chosen before one is retained
shows the picker and sends nothing until a pair is picked, because the manager refuses a mode with
no text. Nothing is drawn on the canvas — a binding is not drawn there today, and a reference is a
binding.

**A runtime error reaches a client through the node-level `error` event and through `node state`;
a per-param descriptor is echoed only by an op.** That was true before and it stands: the editing
scenario reads a reference's arrival error through `node state`. Whether the inspector should learn
of a per-param runtime error without an op echo is open below.

## Open

- A canvas affordance for references, so a patch's modulation is visible where its cables are.
  Nothing in the model prevents it; it is a UI choice.
- The inspector shows a param's RUNTIME error (a shape error on arrival, an evaluation failure)
  only after the next op echo for that node; the node-level `error` event carries the node's
  derived error, not the param's. A per-param runtime error event is one candidate; not decided.
- `triggers` on a reference shares the expression's one arrival path and is unproved by a
  scenario; the editing scenario proves it on an expression only.
