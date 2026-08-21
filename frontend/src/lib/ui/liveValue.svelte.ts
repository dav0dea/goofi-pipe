/** The echo-suppression decision: which value a live control should display. */
export function displayValue<T>(editing: boolean, source: T, local: T): T {
	return editing ? local : source;
}

/** The live-value handle a control drives from its DOM events: `input` buffers, `commit` sends to
 * the backend, and a backend echo is suppressed between `begin` and `end`. */
export interface LiveValue<T> {
	readonly value: T;
	readonly editing: boolean;
	begin(): void;
	input(v: T): void;
	commit(v: T): void;
	end(): void;
}

/** Wire a control's local edit buffer to a live backend source. */
export function useLiveValue<T>(getSource: () => T, onChange: (v: T) => void): LiveValue<T> {
	let editing = $state(false);
	let edit = $state<T>(getSource());

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
