<!-- Parameters panel — edits the parameters of the bound node. -->
<script lang="ts">
	import type { PanelProps } from 'panelty';
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
