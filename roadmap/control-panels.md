# Control panels: what is deferred

The control panel itself is BUILT (2026-09-05): a group of globals, each carrying a `control`
record, drawn as knobs, sliders and fields by the `control` panel type. `globals.group.element` is
the one spelling an expression uses. What is below was decided with the owner and deliberately not
built in that step.

## The reference-selection flow

Picking a control element for a param today means typing `globals.desk.level` into the expression
editor, or using its completion. The flow the owner asked for, and its decisions:

- Right-clicking a control element offers **copy name** and **select for reference**.
- Right-clicking a node OUTPUT SLOT offers the same two.
- There is ONE selection in the app, not one per kind.
- Right-clicking a param then offers **reference selection**, which writes the right thing for what
  is held: a control element sets `mode: expression` with `globals.<group>.<element>`; an output
  slot sets `mode: reference` with `node.slot`.

Two things this must not become: a second selection state beside the canvas's own, and a
touch-unreachable feature — a long press is the coarse door, and `tests/e2e/tests/touch.spec.ts` is
where that is proved.

## A canvas affordance

Nothing draws a control element on the canvas, so a patch's modulation is invisible where its
cables are. This is the same open item `roadmap/param-sources.md` already carries for a reference,
and the two should be answered together rather than separately.

## What a widget follows, and what is still open there (2026-09-05, later)

A global now carries a `source` and the manager writes it from a producer's frames; `learn` in
the widget's inspector binds the first number that moves. Open beside it:

- A source naming a node that is not there is silent: the follower has nothing to read and says
  nothing. A param's reference carries an `error` for the same case; the global's should too.
- The source is chosen through the reference picker. The owner floated dragging a MIDI node off
  the canvas onto the field; nothing prevents it, and the picker is the door until then.
- A double-click renames a widget's label; on touch the inspector's name field is the door, and
  a double-tap is unproved.

## Not going to happen

`cl()` was proposed and dropped. A control element IS a global, so `globals.group.element` is its
one spelling and a second namespace would have been a second owner of one idea.
