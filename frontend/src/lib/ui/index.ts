/**
 * `$lib/ui` — the composable in-house primitive library (sub-project P).
 *
 * The puzzle pieces the inspector (N), the migration (M), and future interfaces
 * assemble. Grows one task at a time; import primitives from here, not by file.
 */
export { default as Button } from './Button.svelte';
export { default as IconButton } from './IconButton.svelte';
export { variantClass, type ButtonVariant, type ButtonSize } from './variantClass';

// Layout primitives (Task 2).
export { default as Stack } from './Stack.svelte';
export { default as Row } from './Row.svelte';
export { default as ScrollArea } from './ScrollArea.svelte';
export { default as Bar } from './Bar.svelte';
export {
	resolveSpace,
	alignItems,
	justifyContent,
	type SpaceScale,
	type AlignSetting,
	type JustifySetting
} from './layout';

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

// Surfaces / overlays (Task 5): the anchored popover, the centered modal dialog, and the panel
// header shell — plus the pure clamp the Popover positions against.
export { default as Popover } from './Popover.svelte';
export { default as Dialog } from './Dialog.svelte';
export { default as PanelShell } from './PanelShell.svelte';
export { clampToViewport, type AnchorRect, type Size, type Placement } from './clampToViewport';
