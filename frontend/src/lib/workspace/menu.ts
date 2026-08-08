import type { IconName } from '$lib/ui';

/** A single entry in a `ContextMenu`. A bare `{ separator: true }` draws a
 * rule; an item with `items` opens a submenu on hover; otherwise `action`
 * fires on click. */
export interface MenuItem {
	label?: string;
	/** A name from the app's icon set — the menu renders it through `$lib/ui`'s `Icon`, the same
	 * way the bar button for the same command does. A glyph would be a second rendering path. */
	icon?: IconName;
	action?: () => void;
	items?: MenuItem[];
	separator?: boolean;
	disabled?: boolean;
	checked?: boolean;
}
