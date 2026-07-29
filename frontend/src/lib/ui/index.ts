/**
 * `$lib/ui` — the composable in-house primitive library (sub-project P).
 *
 * The puzzle pieces the inspector (N), the migration (M), and future interfaces
 * assemble. Grows one task at a time; import primitives from here, not by file.
 */
export { default as Button } from './Button.svelte';
export { default as IconButton } from './IconButton.svelte';
export {
	variantClass,
	type ButtonVariant,
	type ButtonSize,
	type ButtonDensity
} from './variantClass';

// Layout primitives (Task 2). `Stack`/`Row` were retired here (D-M6): a scoped `display: flex`
// rule already says the same thing in the component that needs it, and 86 of the app's 104 flex
// rules also carry position/overflow/padding a frame prop cannot own — adopting them would have
// meant a bespoke override at nearly every site. What survives is the two that carry real
// behaviour: the slim scroller and the pusher bar.
export { default as ScrollArea } from './ScrollArea.svelte';
export { default as Bar } from './Bar.svelte';

// The Field family — the north-star core (Task 3). Dumb controls (value in / change out) that opt
// into the shared `useLiveValue` echo-suppression latch, composed inside a labelled `Field`.
export { default as Field } from './Field.svelte';
export { default as NumberInput } from './NumberInput.svelte';
export { default as Slider } from './Slider.svelte';
export { default as Select } from './Select.svelte';
export { default as TextInput } from './TextInput.svelte';
export { default as Trigger } from './Trigger.svelte';
export { default as Toggle } from './Toggle.svelte';
export { useLiveValue, displayValue, type LiveValue } from './liveValue.svelte';

// Surfaces / overlays (Task 4): the connected tab bar + the collapse control.
export { default as Tabs } from './Tabs.svelte';
export { default as Disclosure } from './Disclosure.svelte';
export { resolveActive, nextIndex, type TabItem, type ArrowKey } from './tabsState';

// Surfaces / overlays (Task 5): the anchored popover and the centered modal dialog — plus the pure
// clamp the Popover positions against. (`PanelShell` was retired here too, D-M6: its one-header-row
// contract is `Bar`'s, which every real panel already composes directly.)
export { default as Popover } from './Popover.svelte';
export { default as Dialog } from './Dialog.svelte';
export {
	clampToViewport,
	overlayViewport,
	type AnchorRect,
	type Size,
	type Placement
} from './clampToViewport';

// Display primitives (Task 6): the uppercase tone pill (static Badge / pressable Chip), the
// glow-free status dot, and the centred empty-state placeholder.
export { default as Badge, type BadgeTone } from './Badge.svelte';
export { default as Chip } from './Chip.svelte';
export { default as StatusDot, type StatusTone, type StatusDotSize } from './StatusDot.svelte';
export { default as EmptyState } from './EmptyState.svelte';
