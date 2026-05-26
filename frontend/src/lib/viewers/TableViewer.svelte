<script lang="ts">
	import type { DataFrame } from '$lib/codec/decode';

	type Props = { frame: DataFrame };
	const { frame }: Props = $props();

	const table = $derived(frame.data as Record<string, DataFrame>);

	function summarize(v: DataFrame): string {
		if (v.dtype === 'STRING') return String(v.data).slice(0, 120);
		if (v.dtype === 'TABLE') return `{${Object.keys(v.data as object).length} fields}`;
		const arr = v.data as { shape: number[]; values: ArrayLike<number> };
		if (arr.shape.length === 0 || (arr.shape.length === 1 && arr.shape[0] === 1)) {
			return String(Number(arr.values[0]));
		}
		return `array[${arr.shape.join('×')}]`;
	}
</script>

<div class="container" data-testid="table-viewer">
	{#each Object.entries(table) as [k, v] (k)}
		<div class="row">
			<div class="k">{k}</div>
			<div class="v" title={summarize(v)}>{summarize(v)}</div>
		</div>
	{/each}
</div>

<style>
	.container {
		width: 100%;
		height: 100%;
		min-height: 80px;
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding: 6px 4px;
		font-size: 10px;
		font-family: var(--font-mono);
		overflow: auto;
	}
	.row {
		display: grid;
		grid-template-columns: 110px 1fr;
		gap: 8px;
	}
	.k {
		color: var(--text-faint);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.v {
		color: var(--text);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
</style>
