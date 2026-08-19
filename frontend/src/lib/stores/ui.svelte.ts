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

/** Compose the key that names a single (node, slot) pair. */
export function slotKey(node: string, slot: string): string {
	return `${node}|${slot}`;
}

export class UIStore {
	/** Ids of the in-panel editors (the fx multi-line, the file browser) that currently own the
	 * keyboard. A ref-count, not a shared boolean: each registers its own id, so collapsing one
	 * never yanks another's standdown, and a leaked `true` from a field that unmounted mid-edit
	 * can't strand the flag.
	 *
	 * Deliberately a PLAIN Set, NOT `$state`. Both mutators must read the membership to decide
	 * whether to write, and every caller is an `$effect` — so a reactive Set makes each registrant
	 * a dependency of the very thing it writes. With one registrant that still converges (the
	 * membership guard short-circuits the second run), which is why it shipped; with TWO it does
	 * not. Svelte runs an effect's teardown BEFORE its body on re-run, so each re-run writes twice
	 * and the two effects invalidate each other until the flush guard throws
	 * `effect_update_depth_exceeded` and leaves the batch dead — every later state change silently
	 * stops applying. The reactive surface is therefore the COUNT alone, which no effect reads. */
	#editors = new Set<string>();
	#openCount = $state(0);

	/** True while any in-panel editor owns the keyboard, so global shortcuts — undo/redo in
	 * particular — stand down. The only external reader of the standdown (undoKeys, AppShell,
	 * `agent/query.ts`), so the set itself stays private. */
	get modalOpen(): boolean {
		return this.#openCount > 0;
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

	/** The input slots an in-flight cable drag is currently near — {@link slotKey} keys, EMPTY
	 * whenever no cable is in flight. An input's name is drawn only on its connector pill, and this
	 * is what asks for it: the editor publishes the near set on each pointermove of a connection
	 * drag, GoofiNode reveals the tags it names. `$state.raw` because the whole Set is replaced (and
	 * only when its membership really changed — see `sameKeys`), never mutated in place. */
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
