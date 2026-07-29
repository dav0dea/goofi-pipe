<!--
  Field — the labelled-control layout primitive (spec §2.2), the north-star core's frame. ONE DOM
  shape replacing the app's 3 incompatible labelled-control layouts: a real `<label>` (`for=`-linked
  to the control it wraps, so clicking the label focuses the control — today only 2 real <label>s
  exist app-wide), an optional `adornment` snippet (where N hangs the `fx` expression binding — it is
  a SIBLING of the label, never inside it, so its button can't steal the label's focus target), and
  the control(s) as `children`. `doc` becomes the hover tooltip.

  The `for=` id is the first LIVE control's, tracked through the field context (see field.ts), so N
  writes `<Field label='cutoff'><Slider/><NumberInput/></Field>` with zero id plumbing. Gap and
  chrome are F tokens exposed as `var(--field-*, <token>)` per-instance hooks. `class` merged,
  `data-testid` (and any other attribute) forwarded via `...rest`.
-->
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
		/** Long-form description — surfaced as the hover tooltip. */
		doc?: string;
		/** Trailing control affordance (N hangs the `fx` binding here). */
		adornment?: Snippet;
		children?: Snippet;
	} = $props();

	// The label points at whichever control is currently alive and first in the field (see field.ts),
	// so an {#if}-swapped control region hands the linkage over instead of orphaning the label.
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
	/* The param name is the primary scan target — bright, weighted, ellipsis'd if long. The label
	   IS clickable (focuses its control), so it carries the pointer cursor to signal that. */
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
	/* A horizontal control row by default (a lone control fills it; a Slider+NumberInput pair sits
	   side by side). Paired controls must be DIRECT Field children — the @container column-flip below
	   restacks `.ui-field-control`'s direct children, so a wrapper (e.g. a Row) around them defeats it. */
	.ui-field-control {
		display: flex;
		align-items: center;
		gap: var(--space-4);
		min-width: 0;
	}
	/* When the enclosing query container (the panel body, or any `container-type` ancestor) is too
	   narrow to seat a paired control side by side, stack the controls into one column. 240px is the
	   width below which a Slider + NumberInput pair stops fitting comfortably — a structural threshold
	   (allowed as literal px, like the F `clamp()` breakpoints), not a themeable token. */
	@container (max-width: 240px) {
		.ui-field-control {
			flex-direction: column;
			align-items: stretch;
		}
	}
</style>
