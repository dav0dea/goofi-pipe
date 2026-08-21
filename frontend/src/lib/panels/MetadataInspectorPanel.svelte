<!-- Metadata panel — shows the incoming Data.meta of the bound node. -->
<script lang="ts">
	import type { PanelProps } from 'panelty';
	import type { NodeInstanceInfo } from '$lib/api/control';
	import NodeLinkedPanel from './NodeLinkedPanel.svelte';
	import MetadataPanel from '$lib/editor/MetadataPanel.svelte';
	import { asStateObject } from 'panelty';
	import { workspace } from 'panelty';
	import { ScrollArea, Select } from '$lib/ui';

	let props: PanelProps = $props();
	const ws = workspace();

	function st(): Record<string, unknown> {
		return asStateObject(props.state);
	}
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
				density="chrome"
				value={curSlot(node) ?? ''}
				onChange={(v) => ws.setPanelSlot(props.panelId, v)}
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
