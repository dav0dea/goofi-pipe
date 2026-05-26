/**
 * Cross-component UI state — small enough to live alongside the graph
 * store but kept separate so re-renders are scoped.
 */
class UIStore {
	/** Which nodes have their viewer panel expanded. */
	expanded = $state<Record<string, boolean>>({});

	toggleExpanded(name: string): void {
		this.expanded = { ...this.expanded, [name]: !this.expanded[name] };
	}

	isExpanded(name: string): boolean {
		return Boolean(this.expanded[name]);
	}
}

let _ui: UIStore | null = null;
export function ui(): UIStore {
	if (!_ui) _ui = new UIStore();
	return _ui;
}
