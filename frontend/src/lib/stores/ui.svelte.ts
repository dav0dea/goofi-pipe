/** Cross-component UI state, kept out of the graph store so re-renders stay scoped. */
export type SlotClickSeed = {
	node: string;
	slot: string;
	dtype: string;
	/** `'source'` when the user clicked an output port; `'target'` for inputs. */
	side: 'source' | 'target';
	clientX: number;
	clientY: number;
};

/** Compose the key that names a single (node, slot) pair. */
export function slotKey(node: string, slot: string): string {
	return `${node}|${slot}`;
}

export class UIStore {
	/** Ids of the in-panel editors that own the keyboard. NOT `$state`: every registrant is an
	 * `$effect`, so a reactive Set makes each a dependency of what it writes — the count is. */
	#editors = new Set<string>();
	#openCount = $state(0);

	/** True while any in-panel editor owns the keyboard, so global shortcuts stand down. */
	get modalOpen(): boolean {
		return this.#openCount > 0;
	}

	/** Bubbled-up "user clicked an unconnected port" intent, cleared by the consumer. */
	pendingSlotClick = $state<SlotClickSeed | null>(null);

	/** Node being dragged out of an editor to link into a panel, or null. */
	nodeDrag = $state<string | null>(null);

	/** Id of the linkable panel the dragged node is over, or null. */
	nodeDragTarget = $state<string | null>(null);

	/** Input slots an in-flight cable drag is near ({@link slotKey} keys); replaced, never mutated. */
	cableNear = $state.raw<ReadonlySet<string>>(new Set());

	setCableNear(keys: ReadonlySet<string>): void {
		this.cableNear = keys;
	}

	isCableNear(node: string, slot: string): boolean {
		return this.cableNear.has(slotKey(node, slot));
	}

	/** Register an open in-panel editor by a stable id (idempotent). */
	openEditor(id: string): void {
		if (this.#editors.has(id)) return;
		this.#editors.add(id);
		this.#openCount = this.#editors.size;
	}

	/** Unregister an editor when it collapses or unmounts (idempotent). */
	closeEditor(id: string): void {
		if (!this.#editors.delete(id)) return;
		this.#openCount = this.#editors.size;
	}

	requestSlotClick(seed: SlotClickSeed): void {
		this.pendingSlotClick = seed;
	}

	consumeSlotClick(): SlotClickSeed | null {
		const seed = this.pendingSlotClick;
		this.pendingSlotClick = null;
		return seed;
	}
}

let _ui: UIStore | null = null;
export function ui(): UIStore {
	if (!_ui) _ui = new UIStore();
	return _ui;
}
