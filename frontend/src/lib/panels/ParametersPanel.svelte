<!-- Parameters panel — edits the parameters of the node dragged into it.
     The group tabs, header-less, are owned by ParamForm (D-N6); this panel just
     supplies the NodeLinkedPanel chrome (linkbar + drag-in + empty state). -->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import NodeLinkedPanel from './NodeLinkedPanel.svelte';
	import ParamForm from '$lib/inspector/ParamForm.svelte';

	let props: PanelProps = $props();
</script>

<NodeLinkedPanel {...props} label="parameters">
	{#snippet content(node)}
		<div class="scroll">
			<ParamForm {node} showHeader={false} />
		</div>
	{/snippet}
</NodeLinkedPanel>

<style>
	.scroll {
		height: 100%;
		overflow-y: auto;
		min-height: 0;
	}
</style>
