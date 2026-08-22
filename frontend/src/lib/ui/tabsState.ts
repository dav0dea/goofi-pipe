/**
 * Pure decisions for the `Tabs` connected bar.
 *
 * `resolveActive` decides which tab reads as active given the `active` prop (falling back to the
 * first when it is unset or names a stale id), and `nextIndex` is the arrow-key navigation model.
 * Kept pure + unit-tested so the component is thin glue and the selection surface is one SSOT —
 * never re-derived (and drifting) inside the render path.
 */

/** A single tab: a stable `id` (what `onSelect` reports) and a human `label`. */
export interface TabItem {
	id: string;
	label: string;
}

/** The keys the tablist navigates with — a horizontal WAI-ARIA tablist (Left/Right + Home/End). */
export type ArrowKey = 'ArrowRight' | 'ArrowLeft' | 'Home' | 'End';

/**
 * Which tab id reads as active: the given `active` when it names a real item, else the first item
 * (so an unset or stale `active` still resolves to a shown tab). `undefined` only when empty.
 */
export function resolveActive(
	items: readonly TabItem[],
	active: string | undefined
): string | undefined {
	if (items.length === 0) return undefined;
	if (active !== undefined && items.some((it) => it.id === active)) return active;
	return items[0].id;
}

/**
 * The index an arrow key moves the selection to, from `current` (or `-1` when nothing is selected).
 * Left/Right step one and wrap; Home/End jump to the ends. Returns `-1` when there are no tabs.
 */
export function nextIndex(current: number, count: number, key: ArrowKey): number {
	if (count <= 0) return -1;
	switch (key) {
		case 'Home':
			return 0;
		case 'End':
			return count - 1;
		case 'ArrowRight':
			return current < 0 ? 0 : (current + 1) % count;
		case 'ArrowLeft':
			return current < 0 ? count - 1 : (current - 1 + count) % count;
	}
}
