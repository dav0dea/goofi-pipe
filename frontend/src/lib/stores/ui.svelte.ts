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

export class UIStore {
	/** Per-slot expand state. Keys are `${node}|${slot}`; entries are the
	 * explicit user-set values. Slots without an entry default to expanded
	 * — viewers should be visible the moment a node arrives, and the user
	 * can collapse individual slots independently.
	 */
	expanded = $state<Record<string, boolean>>({});

	/** Ids of the in-panel editors (the fx multi-line) that currently own the
	 * keyboard. A ref-count, not a shared boolean: each open editor registers
	 * its own id, so collapsing one never yanks another's standdown, and a leaked
	 * `true` from a field that unmounted mid-edit can't strand the flag. Reassigned
	 * (never mutated in place) so `modalOpen` tracks it reactively. */
	openEditors = $state(new Set<string>());

	/** True while any in-panel editor owns the keyboard, so global shortcuts —
	 * undo/redo in particular — stand down. Derived from `openEditors` so the two
	 * readers (undoKeys, AppShell) keep reading `.modalOpen` unchanged. */
	get modalOpen(): boolean {
		return this.openEditors.size > 0;
	}

	/** Bubbled-up "user clicked an unconnected port" intent. NodeEditorPanel.svelte
	 * watches this via $effect and pops the add-node menu pre-seeded for
	 * auto-link. Cleared by the consumer once handled. */
	pendingSlotClick = $state<SlotClickSeed | null>(null);

	/** Name of the node currently being dragged out of an editor to link into a
	 * Parameters / Viewer / Metadata panel (set while a SvelteFlow node drag is
	 * over — or heading toward — a linkable panel). Null otherwise. Drives the
	 * "droppable" outline those panels show. */
	nodeDrag = $state<string | null>(null);

	/** Id of the linkable panel the dragged node is currently over, so that
	 * panel can show the active drop highlight. Null when over none. */
	nodeDragTarget = $state<string | null>(null);

	/** Register an open in-panel editor by a stable id (idempotent). */
	openEditor(id: string): void {
		if (this.openEditors.has(id)) return;
		const next = new Set(this.openEditors);
		next.add(id);
		this.openEditors = next;
	}

	/** Unregister an editor when it collapses or unmounts (idempotent). */
	closeEditor(id: string): void {
		if (!this.openEditors.has(id)) return;
		const next = new Set(this.openEditors);
		next.delete(id);
		this.openEditors = next;
	}

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
