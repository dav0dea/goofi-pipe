# Group-drag of panels

Select several panels and drag them to a new place, keeping their arrangement.

## Where the work lives

The panel system is **panelty** (`dav0dea/panelty`), an npm dependency with a repo of its own. The
tree lives in goofi's document; panelty raises an intent and `LayoutHost` turns each one into a
single manager op. So this feature is a release of the package plus one op mapping here, never a
patch in this tree.

## What is already in its favour

The layout is a flat, id-keyed document root, so moving a set of panels is a set of key writes
rather than a tree surgery. The panel system is frozen UX for restyling, but this is a new gesture
rather than a redesign of an existing one.

## The hard part

**Drop semantics.** A single panel drops into one slot. A selection has an internal arrangement, and
the target has its own — so a drop must say what happens to the shape being carried: does it keep
its split ratios, does it flatten, does it become one tab group? None of the three is obviously
right, and the answer decides the whole gesture.

## Needs

- A multi-select gesture that does not collide with the drag itself, on a touch device as well.
- A drag preview that shows the carried arrangement, because a drop with an unclear outcome is worse
  than no drop.
- ONE op, so a group-drag is one undo step.
