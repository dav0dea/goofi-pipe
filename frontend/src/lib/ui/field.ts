/**
 * Field ⇄ control association (spec §2.2).
 *
 * `<Field>` renders a real `<label for={id}>`; the control it wraps claims that `id` and sets
 * it on its focusable element, so clicking the label focuses the control (native `for=` linkage,
 * no id plumbing at the call site). The claim is one-shot: only the FIRST control in a Field takes
 * the id, so a Field wrapping two controls (`<Slider>` + `<NumberInput>`) yields no duplicate ids
 * and the label focuses one of them.
 *
 * `getContext`/`setContext` run during component init, so both helpers are called synchronously
 * from a component `<script>` (Field before its children, controls as they mount).
 */
import { getContext, setContext } from 'svelte';

const FIELD_KEY = Symbol('ui-field-control');

interface FieldControlContext {
	/** Return the label's `for` id to the first caller, `undefined` to every later one. */
	claim(): string | undefined;
}

/** Called by `<Field>` to publish its label id to the control(s) it wraps. */
export function provideFieldControlId(id: string): void {
	let claimed = false;
	setContext<FieldControlContext>(FIELD_KEY, {
		claim() {
			if (claimed) return undefined;
			claimed = true;
			return id;
		}
	});
}

/**
 * Called by a control to associate with an enclosing `<Field>`'s label. Returns the id to set on
 * the control's focusable element, or `undefined` when standalone (no Field) or already claimed.
 */
export function claimFieldControlId(): string | undefined {
	return getContext<FieldControlContext>(FIELD_KEY)?.claim();
}
