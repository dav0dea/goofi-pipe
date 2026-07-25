<!-- Metadata panel — shows the incoming Data.meta of the node dragged into it.
     The slot picker lives in the panel header bar (next to the node name).
     Linking + empty state are handled by NodeLinkedPanel. -->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import type { NodeInstanceInfo } from '$lib/api/control';
	import NodeLinkedPanel from './NodeLinkedPanel.svelte';
	import MetadataPanel from '$lib/editor/MetadataPanel.svelte';
	import { asStateObject } from '$lib/workspace/panelState';
	import { ScrollArea, Select } from '$lib/ui';

	let props: PanelProps = $props();

	function st(): Record<string, unknown> {
		return asStateObject(props.state);
	}
	// Selected slot persists in the panel state alongside the linked node.
	function curSlot(node: NodeInstanceInfo): string | null {
		const names = Object.keys(node.output_slots ?? {});
		const cur = st().slot;
		return typeof cur === 'string' && node.output_slots[cur] ? cur : (names[0] ?? null);
	}
</script>

<NodeLinkedPanel {...props} label="metadata">
	{#snippet controls(node)}
		{@const slots = Object.keys(node.output_slots ?? {})}
		{#if slots.length > 0}
			<Select
				value={curSlot(node) ?? ''}
				onChange={(v) => props.setState({ ...st(), slot: v })}
				options={slots}
				labels={node.slot_labels}
				data-testid="metadata-slot"
			/>
		{/if}
	{/snippet}

	{#snippet content(node)}
		<ScrollArea>
			<MetadataPanel {node} showHeader={false} slotName={curSlot(node)} />
		</ScrollArea>
	{/snippet}
</NodeLinkedPanel>
