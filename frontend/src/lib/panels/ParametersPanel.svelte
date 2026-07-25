<!-- Parameters panel — edits the parameters of the node dragged into it.
     The group tabs, header-less, are owned by ParamForm (D-N6); this panel just
     supplies the NodeLinkedPanel chrome (linkbar + drag-in + empty state). -->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import NodeLinkedPanel from './NodeLinkedPanel.svelte';
	import ParamForm from '$lib/inspector/ParamForm.svelte';
	import { ScrollArea } from '$lib/ui';

	let props: PanelProps = $props();
</script>

<NodeLinkedPanel {...props} label="parameters">
	{#snippet content(node)}
		<ScrollArea>
			<ParamForm {node} showHeader={false} />
		</ScrollArea>
	{/snippet}
</NodeLinkedPanel>
