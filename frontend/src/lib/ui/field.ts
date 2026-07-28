/**
 * Field ⇄ control association (spec §2.2).
 *
 * `<Field>` renders a real `<label for={id}>`; the control it wraps sets that same `id` on its
 * focusable element, so clicking the label focuses the control (native `for=` linkage, no id
 * plumbing at the call site).
 *
 * The linkage is a LIVE registration, not a one-shot claim: each control registers its own minted
 * id for as long as it is mounted, and the label points at the OLDEST live registrant. That makes
 * both shapes correct by construction —
 *   · a Field wrapping two controls (`<Slider>` + `<NumberInput>`) labels the first, and the two
 *     ids stay distinct;
 *   · a Field whose control region is an `{#if}` chain (the inspector's fx toggle, its ⤢ expand)
 *     hands the label to whichever control is actually alive. Svelte mounts an `{#if}`'s incoming
 *     branch BEFORE tearing the outgoing one down, so the newcomer queues behind the control it
 *     replaces and inherits the label the moment that one is destroyed.
 * A control that names itself (a raw `<textarea aria-label=…>`) simply never registers.
 *
 * `getContext`/`setContext`/`onDestroy` all run during component init, so both helpers are called
 * synchronously from a component `<script>` (Field before its children, controls as they mount).
 */
import { getContext, onDestroy, setContext } from 'svelte';

const FIELD_KEY = Symbol('ui-field-control');

interface FieldControlContext {
	register(id: string): void;
	unregister(id: string): void;
}

/**
 * Called by `<Field>`. `onLabelFor` is invoked with the id the label should point at — the oldest
 * live control's, or `undefined` while the field wraps no registering control.
 */
export function provideFieldControlId(onLabelFor: (id: string | undefined) => void): void {
	// Plain (non-reactive) bookkeeping in mount order; the ONE reactive surface is the caller's,
	// written only from here, never read — so no effect can become its own dependency.
	const live: string[] = [];
	setContext<FieldControlContext>(FIELD_KEY, {
		register(id) {
			live.push(id);
			onLabelFor(live[0]);
		},
		unregister(id) {
			const i = live.indexOf(id);
			if (i >= 0) live.splice(i, 1);
			onLabelFor(live[0]);
		}
	});
}

/**
 * Called by a control with its own minted id (`$props.id()`), to associate with an enclosing
 * `<Field>`'s label. Returns the id to set on the control's focusable element, or `undefined` when
 * standalone (no Field) — in which case the control renders without an id, as before.
 */
export function claimFieldControlId(id: string): string | undefined {
	const ctx = getContext<FieldControlContext>(FIELD_KEY);
	if (!ctx) return undefined;
	ctx.register(id);
	onDestroy(() => ctx.unregister(id));
	return id;
}
