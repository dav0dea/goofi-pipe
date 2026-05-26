<script lang="ts">
	import type { ArrayData } from '$lib/codec/decode';

	type Props = { arraySpec: ArrayData };
	const { arraySpec }: Props = $props();

	const stats = $derived.by(() => {
		const v = arraySpec.values as ArrayLike<number>;
		if (v.length === 0) return null;
		let mn = Infinity;
		let mx = -Infinity;
		let sum = 0;
		let n = 0;
		for (let i = 0; i < v.length; i++) {
			const x = Number(v[i]);
			if (!Number.isFinite(x)) continue;
			if (x < mn) mn = x;
			if (x > mx) mx = x;
			sum += x;
			n++;
		}
		if (n === 0) return null;
		return { min: mn, max: mx, mean: sum / n, n };
	});
</script>

<div class="fallback" data-testid="highdim-fallback">
	<div class="title">Cannot render this shape</div>
	<div class="line">
		shape: <code>[{arraySpec.shape.join(', ')}]</code>
	</div>
	<div class="line">
		dtype: <code>{arraySpec.dtype}</code>
	</div>
	{#if stats}
		<div class="line">min: <code>{stats.min.toPrecision(4)}</code></div>
		<div class="line">max: <code>{stats.max.toPrecision(4)}</code></div>
		<div class="line">mean: <code>{stats.mean.toPrecision(4)}</code></div>
	{:else}
		<div class="line">stats: no finite values</div>
	{/if}
</div>

<style>
	.fallback {
		font-family: var(--font-mono);
		font-size: 10px;
		color: var(--text-dim);
		padding: 6px 8px;
		display: flex;
		flex-direction: column;
		gap: 2px;
		align-self: stretch;
	}
	.title {
		color: var(--warning);
		font-weight: 600;
		margin-bottom: 4px;
	}
	code {
		color: var(--text);
	}
</style>
