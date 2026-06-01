<!-- Metadata panel — the incoming Data.meta inspector for the selected node.
     Adapter over the existing editor/MetadataPanel, fed from selection. -->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import MetadataPanel from '$lib/editor/MetadataPanel.svelte';
	import { selection } from '$lib/stores/selection.svelte';

	// eslint-disable-next-line @typescript-eslint/no-unused-vars
	let {}: PanelProps = $props();
	const sel = selection();
</script>

<div class="wrap">
	{#if sel.selectedNode}
		<MetadataPanel node={sel.selectedNode} />
	{:else}
		<div class="hint">Select a node to inspect its output metadata</div>
	{/if}
</div>

<style>
	.wrap {
		height: 100%;
		overflow-y: auto;
		min-height: 0;
	}
	.hint {
		display: grid;
		place-items: center;
		height: 100%;
		padding: 16px;
		text-align: center;
		color: var(--text-faint);
	}
</style>
