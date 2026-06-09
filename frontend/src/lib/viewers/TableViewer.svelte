<script lang="ts">
	import type { DataFrame } from '$lib/codec/decode';
	import type { SettingsMap } from './viewerSettings.svelte';

	type Props = { frame: DataFrame; settings?: SettingsMap };
	const { frame, settings = {} }: Props = $props();

	const decimals = $derived(Math.max(0, Math.min(10, Number(settings.decimals ?? 3))));
	const table = $derived(frame.data as Record<string, DataFrame>);

	function fmtNum(n: number): string {
		return Number.isFinite(n) ? n.toFixed(decimals) : String(n);
	}

	function summarize(v: DataFrame): string {
		if (v.dtype === 'STRING') return String(v.data).slice(0, 120);
		if (v.dtype === 'TABLE') return `{${Object.keys(v.data as object).length} fields}`;
		const arr = v.data as { shape: number[]; values: ArrayLike<number> };
		if (arr.shape.length === 0 || (arr.shape.length === 1 && arr.shape[0] === 1)) {
			return fmtNum(Number(arr.values[0]));
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
