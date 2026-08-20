/** A single entry in a `ContextMenu`. A bare `{ separator: true }` draws a
 * rule; an item with `items` opens a submenu on hover; otherwise `action`
 * fires on click. */
export interface MenuItem {
	label?: string;
	/** An icon name for the one renderer — a chrome glyph or one the consumer registered, the same
	 * way the bar button for the same command names it. A glyph would be a second rendering path. */
	icon?: string;
	action?: () => void;
	items?: MenuItem[];
	separator?: boolean;
	disabled?: boolean;
	checked?: boolean;
}
