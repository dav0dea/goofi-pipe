<!-- Field — the labelled-control frame: a real `<label>` for=-linked to the first live control
     (see field.ts), plus an `adornment` that is a SIBLING of the label, never inside it. -->
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
	<div class="ui-field-head">
		<label class="ui-field-label" for={controlId}>{label}</label>
		{#if adornment}
			<span class="ui-field-adornment">{@render adornment()}</span>
		{/if}
	</div>
	<div class="ui-field-control">{@render children?.()}</div>
</div>

<style>
	.ui-field {
		display: flex;
		flex-direction: column;
		gap: var(--field-gap, var(--space-3));
		min-width: 0;
		font-size: var(--fs-small);
	}
	.ui-field-head {
		display: flex;
		align-items: center;
		gap: var(--space-4);
		min-width: 0;
	}
	.ui-field-label {
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		color: var(--text);
		font-weight: 600;
		letter-spacing: 0.01em;
		cursor: pointer;
	}
	.ui-field-adornment {
		flex-shrink: 0;
		display: inline-flex;
		align-items: center;
		gap: var(--space-2);
	}
	/* Paired controls must be DIRECT children: the @container flip below restacks direct children. */
	.ui-field-control {
		display: flex;
		align-items: center;
		gap: var(--space-4);
		min-width: 0;
	}
	/* 240px is where a Slider + NumberInput pair stops fitting: a structural threshold, not a token. */
	@container (max-width: 240px) {
		.ui-field-control {
			flex-direction: column;
			align-items: stretch;
		}
	}
</style>
