<!-- Metadata panel — shows the incoming Data.meta of the node dragged into it.
     Linking + empty state are handled by NodeLinkedPanel. -->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import NodeLinkedPanel from './NodeLinkedPanel.svelte';
	import MetadataPanel from '$lib/editor/MetadataPanel.svelte';

	let props: PanelProps = $props();
</script>

<NodeLinkedPanel {...props} label="metadata">
	{#snippet content(node)}
		<div class="scroll"><MetadataPanel {node} /></div>
	{/snippet}
</NodeLinkedPanel>

<style>
	.scroll {
		height: 100%;
		overflow-y: auto;
		min-height: 0;
	}
</style>
