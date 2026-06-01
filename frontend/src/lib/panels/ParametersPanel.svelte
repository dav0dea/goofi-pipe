<!-- Parameters panel — edits the parameters of the node dragged into it.
     Linking + empty state are handled by NodeLinkedPanel. -->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import NodeLinkedPanel from './NodeLinkedPanel.svelte';
	import ParamPanel from '$lib/params/ParamPanel.svelte';

	let props: PanelProps = $props();
</script>

<NodeLinkedPanel {...props} label="parameters">
	{#snippet content(node)}
		<div class="scroll"><ParamPanel {node} showHeader={false} /></div>
	{/snippet}
</NodeLinkedPanel>

<style>
	.scroll {
		height: 100%;
		overflow-y: auto;
		min-height: 0;
	}
</style>
