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

	/** Per-node CSS translation applied during a snapped drag. Editor writes
	 * the snap delta here each `onnodedrag` tick; GoofiNode reads the value
	 * and overlays a `transform: translate(...)` so the node visibly tracks
	 * the snap guides instead of the raw mouse position. Cleared on dragstop. */
	dragSnap = $state<Record<string, { dx: number; dy: number }>>({});

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

	isSlotExpanded(node: string, slot: string): boolean {
		const k = slotKey(node, slot);
		if (Object.prototype.hasOwnProperty.call(this.expanded, k)) return this.expanded[k];
		return true; // default: visible
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
