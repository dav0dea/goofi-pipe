<!-- Field — the labelled-control frame: a real `<label>` for=-linked to the first live control
     (see field.ts), plus an `adornment` that is a SIBLING of the label, never inside it. The row
     wraps on its OWN width rather than on a breakpoint, so one field per line is the resting
     shape and the label steps above the value only where the two cannot share a line. -->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';
	import { provideFieldControlId } from './field';

	let {
		label,
		doc,
		adornment,
		class: klass = '',
		children,
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		label: string;
		/** Long-form description, surfaced as the hover tooltip. */
		doc?: string;
		/** Trailing control affordance. */
		adornment?: Snippet;
		children?: Snippet;
	} = $props();

	let controlId = $state<string | undefined>(undefined);
	provideFieldControlId((id) => (controlId = id));
</script>

<div {...rest} class={`ui-field ${klass}`.trim()} title={doc ?? rest.title}>
	<label class="ui-field-label" for={controlId}>{label}</label>
	<div class="ui-field-value">
		<div class="ui-field-control">{@render children?.()}</div>
		{#if adornment}
			<span class="ui-field-adornment">{@render adornment()}</span>
		{/if}
	</div>
</div>

<style>
	.ui-field {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		column-gap: var(--space-4);
		row-gap: var(--field-gap, var(--space-3));
		min-width: 0;
		font-size: var(--fs-small);
	}
	.ui-field-label {
		flex: 0 1 auto;
		min-width: 4rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		color: var(--text);
		font-weight: 600;
		letter-spacing: 0.01em;
		cursor: pointer;
	}
	/* ONE flex item, so a field too narrow for a single line drops the whole value under the label
	   instead of orphaning the adornment on a line of its own. Its basis IS the wrap threshold, and
	   it sits ABOVE the stack threshold below so the two stages cannot both leave a control squeezed. */
	.ui-field-value {
		flex: 1 1 18rem;
		min-width: 0;
		display: flex;
		align-items: center;
		gap: var(--space-4);
	}
	/* Paired controls must be DIRECT children: the @container flip below restacks direct children. */
	.ui-field-control {
		flex: 1 1 auto;
		display: flex;
		align-items: center;
		gap: var(--space-4);
		min-width: 0;
	}
	.ui-field-adornment {
		flex-shrink: 0;
		display: inline-flex;
		align-items: center;
		gap: var(--space-2);
	}
	/* Where a Slider + NumberInput pair stops fitting beside an adornment: a structural threshold,
	   not a token — and in `rem`, the unit the controls it measures are drawn in. */
	@container (max-width: 20rem) {
		.ui-field-control {
			flex-direction: column;
			align-items: stretch;
		}
	}
</style>
