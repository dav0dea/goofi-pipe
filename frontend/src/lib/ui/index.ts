/**
 * `$lib/ui` — the composable in-house primitive library (sub-project P).
 *
 * The puzzle pieces the inspector (N), the migration (M), and future interfaces
 * assemble. Grows one task at a time; import primitives from here, not by file.
 */
// The chrome primitives and the icon renderer are the PANEL SYSTEM's, and are taken back here so
// the app imports every primitive from one place. A panel's content sits inside that chrome, so a
// second button beside it would be a foreign control two pixels from a native one. `ICONS` is the
// app's own geometry, which it registers into that one renderer at startup.
export { default as Button } from '$lib/workspace/ui/Button.svelte';
export { default as IconButton } from '$lib/workspace/ui/IconButton.svelte';
export { default as Icon } from '$lib/workspace/ui/Icon.svelte';
export { type ButtonVariant, type ButtonSize } from '$lib/workspace/ui/variantClass';
export { ICONS, type IconName } from './icons';

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
export { MODE_ATTRS } from './inputMode';
export { default as Trigger } from './Trigger.svelte';
export { default as Toggle } from './Toggle.svelte';

// Surfaces / overlays (Task 4): the connected tab bar + the collapse control. The tab bar is the
// panel system's — the workspace strip and the inspector's param groups are one component, and it
// is the strip that the panel drag system is one half of.
export { default as Tabs } from '$lib/workspace/ui/Tabs.svelte';
export { type TabItem } from '$lib/workspace/ui/tabsState';
export { default as Disclosure } from './Disclosure.svelte';

// Surfaces / overlays (Task 5): the anchored popover and the centered modal dialog. (`PanelShell`
// was retired here too, D-M6: its one-header-row contract is `Bar`'s, which every real panel
// already composes directly.)
export { default as Popover } from './Popover.svelte';
export { default as Dialog } from './Dialog.svelte';

// `beginDrag`, `clampToViewport` and `portal` used to be re-exported here. They are pure DOM
// helpers with no design in them, so they live one layer DOWN, in `$lib/gesture` — which is what
// lets this library stay a leaf while the panel system uses all three.

// Not a primitive, but the one answer to "does this
// keystroke belong to a text editor or to the app", shared by the two keyboard scopes that would
// otherwise each keep their own list of editable tags — and one of them would miss the next
// contenteditable, as both did before X.
export { isTextEditingTarget } from './textEditing';

// Display primitives (Task 6): the uppercase tone pill (static Badge / pressable Chip), the
// glow-free status dot, and the centred empty-state placeholder.
export { default as Badge, type BadgeTone } from './Badge.svelte';
export { default as Chip } from './Chip.svelte';
export { default as StatusDot, type StatusTone, type StatusDotSize } from './StatusDot.svelte';
export { default as EmptyState } from './EmptyState.svelte';
export { default as ChoiceGrid, type Choice } from './ChoiceGrid.svelte';
