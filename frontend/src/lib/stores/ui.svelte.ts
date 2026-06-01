/**
 * Cross-component UI state — small enough to live alongside the graph
 * store but kept separate so re-renders are scoped.
 */
export type SlotClickSeed = {
	node: string;
	slot: string;
	dtype: string;
	/** `'source'` when the user clicked an output port; `'target'` for inputs. */
	side: 'source' | 'target';
	/** Pointer position when the click happened — used to position the menu. */
	clientX: number;
	clientY: number;
};

/** Compose the storage key for a single slot's expand state. */
export function slotKey(node: string, slot: string): string {
	return `${node}|${slot}`;
}

class UIStore {
	/** Per-slot expand state. Keys are `${node}|${slot}`; entries are the
	 * explicit user-set values. Slots without an entry default to expanded
	 * — viewers should be visible the moment a node arrives, and the user
	 * can collapse individual slots independently.
	 */
	expanded = $state<Record<string, boolean>>({});

	/** Bubbled-up "user clicked an unconnected port" intent. Editor.svelte
	 * watches this via $effect and pops the add-node menu pre-seeded for
	 * auto-link. Cleared by the consumer once handled. */
	pendingSlotClick = $state<SlotClickSeed | null>(null);

	/** Name of the node currently being dragged out of an editor (by its grip)
	 * to link into a Parameters / Viewer / Metadata panel. Null when no such
	 * drag is in progress. */
	nodeDrag = $state<string | null>(null);

	requestSlotClick(seed: SlotClickSeed): void {
		this.pendingSlotClick = seed;
	}

	consumeSlotClick(): SlotClickSeed | null {
		const seed = this.pendingSlotClick;
		this.pendingSlotClick = null;
		return seed;
	}

	toggleSlotExpanded(node: string, slot: string): void {
		const k = slotKey(node, slot);
		this.expanded = { ...this.expanded, [k]: !this.isSlotExpanded(node, slot) };
	}

	setSlotExpanded(node: string, slot: string, expanded: boolean): void {
		this.expanded = { ...this.expanded, [slotKey(node, slot)]: expanded };
	}

	isSlotExpanded(node: string, slot: string): boolean {
		const k = slotKey(node, slot);
		if (Object.prototype.hasOwnProperty.call(this.expanded, k)) return this.expanded[k];
		return true; // default: visible
	}

	/** Seed expand state for a freshly-arrived node.
	 *
	 * - If the node carries an explicit `viewers` map (loaded from a saved
	 *   patch), apply each slot's saved `collapsed` flag.
	 * - Otherwise this is a fresh spawn: default to expanded, except when
	 *   the node has 3+ outputs, in which case we start everything collapsed
	 *   to avoid burying the canvas under tall viewers.
	 */
	seedNodeViewers(
		node: string,
		outputSlots: string[],
		saved: Record<string, { collapsed?: boolean }> | undefined
	): void {
		const next = { ...this.expanded };
		const hasSaved = saved && Object.keys(saved).length > 0;
		const defaultCollapsed = !hasSaved && outputSlots.length >= 3;
		for (const slot of outputSlots) {
			const savedFor = saved?.[slot];
			const collapsed = savedFor?.collapsed ?? defaultCollapsed;
			next[slotKey(node, slot)] = !collapsed;
		}
		this.expanded = next;
	}

	/** Drop bookkeeping for every slot of a node that no longer exists. */
	forget(name: string): void {
		const prefix = `${name}|`;
		let changed = false;
		const next = { ...this.expanded };
		for (const k of Object.keys(next)) {
			if (k.startsWith(prefix)) {
				delete next[k];
				changed = true;
			}
		}
		if (changed) this.expanded = next;
	}
}

let _ui: UIStore | null = null;
export function ui(): UIStore {
	if (!_ui) _ui = new UIStore();
	return _ui;
}
