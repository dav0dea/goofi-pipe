/**
 * `useLiveValue` — the shared commit / echo-suppression discipline for the Field controls
 * (spec §3). While the user is actively editing (focused / dragging), backend echoes are
 * suppressed so the value doesn't jump under the cursor; on blur / Enter the edit is committed
 * and syncing resumes. Every live control opts into this rune instead of re-implementing the
 * latch (and re-introducing the bug).
 *
 * The one decision — which value to show — is the PURE `displayValue`, unit-tested in
 * `liveValue.test.ts`. The rune below is the thin Svelte-reactive wrapper (`$state`/`$effect`)
 * that ties it to the live source and the control's `onChange` callback.
 */

/**
 * The echo-suppression decision: which value a live control should display.
 * - idle (`editing === false`) → follow the backend `source` (resume syncing),
 * - editing → keep the in-progress `local` edit (suppress the echo).
 *
 * Pure and generic over the control's value kind, so it is tested once for all controls.
 */
export function displayValue<T>(editing: boolean, source: T, local: T): T {
	return editing ? local : source;
}

/** The live-value handle a control drives from its DOM events. */
export interface LiveValue<T> {
	/** The value to render — echo-suppressed while editing, else the live source. */
	readonly value: T;
	/** Whether an edit is in progress (echoes currently suppressed). */
	readonly editing: boolean;
	/** Enter the editing latch (focus / pointer-down) — start suppressing echoes. */
	begin(): void;
	/** Update the in-progress value WITHOUT committing (typing / buffering). */
	input(v: T): void;
	/** Commit a value to the backend via `onChange` (blur / Enter, or a live drag step). */
	commit(v: T): void;
	/** Release the editing latch WITHOUT committing (pointer-up / blur after live commits). */
	end(): void;
}

/**
 * Wire a control's local edit buffer to a live backend source.
 *
 * @param getSource reactive getter for the backend value (tracked by the derived below).
 * @param onChange  called with each committed value.
 */
export function useLiveValue<T>(getSource: () => T, onChange: (v: T) => void): LiveValue<T> {
	let editing = $state(false);
	let edit = $state<T>(getSource()); // the in-progress buffer; only meaningful while editing

	// The one decision, from the pure function: idle → the live source, editing → the local edit.
	// A derived (not an effect) so there is exactly ONE place the rule lives and no self-dependency.
	const value = $derived(displayValue(editing, getSource(), edit));

	return {
		get value() {
			return value;
		},
		get editing() {
			return editing;
		},
		begin() {
			edit = getSource(); // seed from the source so there's no flash to a stale edit
			editing = true;
		},
		input(v: T) {
			edit = v;
		},
		commit(v: T) {
			edit = v;
			onChange(v);
		},
		end() {
			editing = false;
		}
	};
}
