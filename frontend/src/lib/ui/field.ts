/** Field ⇄ control association: a control registers its minted id while mounted, and the label
 * points at the OLDEST live registrant, so an `{#if}` chain hands the label on as branches swap. */
import { getContext, onDestroy, setContext } from 'svelte';

const FIELD_KEY = Symbol('ui-field-control');

interface FieldControlContext {
	register(id: string): void;
	unregister(id: string): void;
}

/** Called by `<Field>`; `onLabelFor` receives the id the label should point at, or `undefined`. */
export function provideFieldControlId(onLabelFor: (id: string | undefined) => void): void {
	// Plain, non-reactive bookkeeping: nothing here may become its own effect dependency.
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

/** Associates a control's minted id with an enclosing `<Field>`; `undefined` when standalone. */
export function claimFieldControlId(id: string): string | undefined {
	const ctx = getContext<FieldControlContext>(FIELD_KEY);
	if (!ctx) return undefined;
	ctx.register(id);
	onDestroy(() => ctx.unregister(id));
	return id;
}
