/**
 * Cross-component UI state — small enough to live alongside the graph
 * store but kept separate so re-renders are scoped.
 */
class UIStore {
	/** Which nodes have their viewer panel expanded.
	 *
	 * Default policy: a node is expanded unless the user has explicitly
	 * collapsed it. We store the explicit user-set value here; `isExpanded`
	 * falls back to `true` when no entry exists. Goofi3-style "viewers
	 * visible by default", but the ▾/▸ toggle still lets the user hide them.
	 */
	expanded = $state<Record<string, boolean>>({});

	toggleExpanded(name: string): void {
		this.expanded = { ...this.expanded, [name]: !this.isExpanded(name) };
	}

	isExpanded(name: string): boolean {
		if (Object.prototype.hasOwnProperty.call(this.expanded, name)) return this.expanded[name];
		return true; // default: visible
	}

	/** Drop bookkeeping for nodes that no longer exist. */
	forget(name: string): void {
		if (Object.prototype.hasOwnProperty.call(this.expanded, name)) {
			const next = { ...this.expanded };
			delete next[name];
			this.expanded = next;
		}
	}
}

let _ui: UIStore | null = null;
export function ui(): UIStore {
	if (!_ui) _ui = new UIStore();
	return _ui;
}
