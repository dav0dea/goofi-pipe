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
