<!-- EmptyState — a centred "nothing here yet" placeholder; every part is optional. -->
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
