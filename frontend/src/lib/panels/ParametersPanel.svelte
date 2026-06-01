<!-- Parameters panel — the param editor for the currently selected node. Thin
     adapter over the existing ParamPanel, fed from the shared selection store. -->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import ParamPanel from '$lib/params/ParamPanel.svelte';
	import { selection } from '$lib/stores/selection.svelte';

	// eslint-disable-next-line @typescript-eslint/no-unused-vars
	let {}: PanelProps = $props();
	const sel = selection();
</script>

<div class="wrap">
	<ParamPanel node={sel.activeSelectedNode} />
</div>

<style>
	.wrap {
		height: 100%;
		overflow-y: auto;
		min-height: 0;
	}
</style>
