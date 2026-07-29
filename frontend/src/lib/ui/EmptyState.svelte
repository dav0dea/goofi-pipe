<!--
  EmptyState — a centred placeholder (spec §2.5): optional `title` / `hint` snippets stacked and
  centred on both axes for the "nothing here yet" panel state, plus an optional default `children`
  slot for a call-to-action. Every part is optional and the layout collapses gracefully when any is
  absent (a bare <EmptyState /> is a valid, empty centred box). Muted typography (--text-dim title,
  --text-muted hint) so it recedes. `class` merged, `data-testid` (+ any attribute) forwarded.

  There was an `icon` snippet too. In eleven consumers not one passed it — only the gallery did,
  which is the same clause that deleted `Field.hint` at 58f3136: a prop no dispatch channel can
  reach is not an escape hatch, it is a frame nobody needed.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';

	let {
		title,
		hint,
		class: klass = '',
		children,
		...rest
	}: Omit<HTMLAttributes<HTMLDivElement>, 'title'> & {
		title?: Snippet;
		hint?: Snippet;
		children?: Snippet;
	} = $props();
</script>

<div {...rest} class={`ui-empty-state ${klass}`.trim()}>
	{#if title}<div class="ui-empty-title">{@render title()}</div>{/if}
	{#if hint}<div class="ui-empty-hint">{@render hint()}</div>{/if}
	{@render children?.()}
</div>

<style>
	.ui-empty-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: var(--space-4);
		padding: var(--space-8);
		min-width: 0;
		text-align: center;
	}
	.ui-empty-title {
		color: var(--text-dim);
		font-size: var(--fs-strong);
		font-weight: 600;
	}
	.ui-empty-hint {
		max-width: 24rem;
		color: var(--text-muted);
		font-size: var(--fs-small);
	}
</style>
