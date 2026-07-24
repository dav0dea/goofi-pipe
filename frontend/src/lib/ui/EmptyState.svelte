<!--
  EmptyState — a centred placeholder (spec §2.5): optional `icon` / `title` / `hint` snippets stacked
  and centred on both axes for the "nothing here yet" panel state, plus an optional default `children`
  slot for a call-to-action. Every part is optional and the layout collapses gracefully when any is
  absent (a bare <EmptyState /> is a valid, empty centred box). Muted typography (--text-dim title,
  --text-muted icon/hint) so it recedes. `class` merged, `data-testid` (+ any attribute) forwarded.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';

	let {
		icon,
		title,
		hint,
		class: klass = '',
		children,
		...rest
	}: Omit<HTMLAttributes<HTMLDivElement>, 'title'> & {
		icon?: Snippet;
		title?: Snippet;
		hint?: Snippet;
		children?: Snippet;
	} = $props();
</script>

<div {...rest} class={`ui-empty-state ${klass}`.trim()}>
	{#if icon}<div class="ui-empty-icon">{@render icon()}</div>{/if}
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
	.ui-empty-icon {
		color: var(--text-muted);
		font-size: var(--fs-title);
		line-height: 1;
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
