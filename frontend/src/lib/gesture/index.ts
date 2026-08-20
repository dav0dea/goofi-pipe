/**
 * `$lib/gesture` — the leaf-most layer: pure pointer, geometry and measurement helpers that touch
 * the DOM and nothing else. No store, no primitive, no token, no import of any kind outside
 * `svelte/action`.
 *
 * They sit below `$lib/ui` rather than inside it because the panel system needs them and the
 * primitive library must stay a leaf — `Popover` reaching up into `$lib/workspace` for `portal` was
 * the layering inversion this directory removes. A chrome strip's overflow plan is here for the
 * same reason: the app header and a panel header both make it, and neither owns it.
 */
export { portal } from './portal';
export { beginDrag } from './dragGesture';
export { clampToViewport, overlayViewport, MARGIN } from './clampToViewport';
export { createLongPress, HOLD_MS, MOVE_TOLERANCE_PX } from './longPress';
export { createWidthCache, planOverflow, type OverflowItem } from './overflowFit';
