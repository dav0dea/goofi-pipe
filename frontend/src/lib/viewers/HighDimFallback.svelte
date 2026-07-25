<script lang="ts">
	import type { ViewSummary } from './viewMeta';

	type Props = { summary: ViewSummary };
	const { summary }: Props = $props();

	const hasStats = $derived(
		summary.min !== null && summary.max !== null && summary.mean !== null
	);
</script>

<div class="fallback" data-testid="highdim-fallback">
	<div class="title">Cannot render this shape</div>
	<div class="line">
		shape: <code>[{summary.shape.join(', ')}]</code>
	</div>
	<div class="line">
		dtype: <code>{summary.dtype}</code>
	</div>
	{#if hasStats}
		<div class="line">min: <code>{summary.min!.toPrecision(4)}</code></div>
		<div class="line">max: <code>{summary.max!.toPrecision(4)}</code></div>
		<div class="line">mean: <code>{summary.mean!.toPrecision(4)}</code></div>
	{:else}
		<div class="line">stats: no finite values</div>
	{/if}
</div>

<style>
	.fallback {
		font-family: var(--font-mono);
		font-size: var(--fs-micro);
		color: var(--text-dim);
		padding: var(--space-3) var(--space-5);
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
		align-self: stretch;
	}
	.title {
		color: var(--warning);
		font-weight: 600;
		margin-bottom: var(--space-2);
	}
	code {
		color: var(--text);
	}
</style>
