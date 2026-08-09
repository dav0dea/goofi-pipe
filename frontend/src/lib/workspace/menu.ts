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
	/** A stable handle for a row that has more than one door onto it. The header's `Save As…` is
	 * the case: it is a row of the Save split-button's dropdown until the caret spills, and a row
	 * of the ⋯ overflow menu after — so a driver that must reach it at ANY width cannot key on
	 * which menu it is in, and keying on the label alone would break with the wording. */
	testid?: string;
	items?: MenuItem[];
	separator?: boolean;
	disabled?: boolean;
	checked?: boolean;
}
