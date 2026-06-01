/** A single entry in a `ContextMenu`. A bare `{ separator: true }` draws a
 * rule; an item with `items` opens a submenu on hover; otherwise `action`
 * fires on click. */
export interface MenuItem {
	label?: string;
	icon?: string;
	action?: () => void;
	items?: MenuItem[];
	separator?: boolean;
	disabled?: boolean;
	checked?: boolean;
}
