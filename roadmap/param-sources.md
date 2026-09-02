# Param sources: constant, expression, reference

Every param of every node, on every engine, has exactly ONE active source. Decided with the user
on 2026-09-02 while designing the audio engine, because it is what gives that engine control-rate
and audio-rate modulation through one door with no precedence rule between them — and it applies
to the whole system, not to audio. Not built; the first step of the audio program.

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

**A reference is the string `node.slot`, by name**, in the record, the document, the `.gfi` and
the op. A rename rewrites it exactly as it rewrites `nd('name')` in an expression — one rename
mechanism. A deleted producer leaves the reference in place with a binding error and the literal
stands, as an unresolved `nd()` does; undo restores it. Nothing is pruned.

**Node names and slot names are `[A-Za-z][A-Za-z0-9]*` and not a Python keyword.** An expression
reads a slot as an attribute, and the rule is what makes `node.slot` unambiguous with no quoting.
One function in `goofi-core`, beside `is_valid_identifier` (which `globals` keep, underscore
included), enforced at `node add`, at rename, by the Python probe (a bad slot name registers the
type UNAVAILABLE with the name quoted), by every engine's own declaration path the same way, and
by a test over every shipped manifest. `in` is a keyword and therefore not a slot name.

**Typing is by slot kind.** `Float`, `Int` and `Bool` reference an `Array` or an `Audio` output;
`Str` references a `String` output. A mismatch is a bind error at the op, and the literal stands.
On the signal plane a referenced `Array` frame must hold ONE element; a wider frame reports a
shape error through `BindingErrors` on arrival, the literal stands, and an expression with a
reduction is the tool.

**`triggers` applies to expression and reference alike.**

**The op is `node param edit`, grown by one flag**: `reference: string` beside `expression`, and
`mode` accepts `reference`. Giving a reference implies the mode, as giving an expression does;
an empty reference clears it; a mode alone switches among what is retained.

**The document shape is `{value, mode, expr: {source, triggers}, ref: "node.slot"}`**, `expr` and
`ref` present when they have content. The `.gfi` carries the same. No shim for the old `enabled`
shape: the pre-launch policy applies.

**The seam ships the source kind on `BindingView`.** A reference is one `BoundVar::Stream` with no
compiled id; an expression is what it is today. The signal engine subscribes a reference like a
binding var and COPIES the scalar on arrival — no evaluator, no GIL, none of the `codes` mutex.
The audio engine makes a same-engine reference a plan edge (per sample, per channel: a 4-channel
gate referenced into an envelope is four voices) and lands a foreign reference or an expression as
an atomic at control rate.

**The inspector's fx badge becomes a three-way toggle.** Reference mode is two fields side by
side, node then slot, each the expression editor in a picker configuration: no language, no
highlighting, one completion source over the catalogue the expression editor already reads.
Choosing a node enables the slot field; the pair commits as one `node.slot` in one op. Nothing is
drawn on the canvas — a binding is not drawn there today, and a reference is a binding.

## Open

- A canvas affordance for references, so a patch's modulation is visible where its cables are.
  Nothing in the model prevents it; it is a UI choice.
