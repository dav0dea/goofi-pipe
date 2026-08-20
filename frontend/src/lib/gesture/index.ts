/**
 * `$lib/gesture` — the leaf-most layer: pure pointer and geometry helpers that touch the DOM and
 * nothing else. No store, no primitive, no token, no import of any kind outside `svelte/action`.
 *
 * They sit below `$lib/ui` rather than inside it because the panel system needs all three and the
 * primitive library must stay a leaf — `Popover` reaching up into `$lib/workspace` for `portal` was
 * the layering inversion this directory removes.
 */
export { portal } from './portal';
export { beginDrag } from './dragGesture';
export { clampToViewport, overlayViewport, MARGIN } from './clampToViewport';
