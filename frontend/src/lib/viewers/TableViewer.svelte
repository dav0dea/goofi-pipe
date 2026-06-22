<script lang="ts">
	import type { DataFrame } from '$lib/codec/decode';
	import type { SettingsMap } from './settingsSchema';
	import TableTree from './TableTree.svelte';

	type Props = { frame: DataFrame; settings?: SettingsMap };
	const { frame, settings = {} }: Props = $props();

	const decimals = $derived(Math.max(0, Math.min(10, Number(settings.decimals ?? 3))));
	const table = $derived(frame.data as Record<string, DataFrame>);
</script>

<div class="container" data-testid="table-viewer">
	{#each Object.entries(table) as [k, v] (k)}
		<TableTree name={k} frame={v} {decimals} />
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
</style>
