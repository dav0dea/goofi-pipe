<!-- Metadata panel — shows the incoming Data.meta of the node dragged into it.
     The slot picker lives in the panel header bar (next to the node name).
     Linking + empty state are handled by NodeLinkedPanel. -->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import type { NodeInstanceInfo } from '$lib/api/control';
	import NodeLinkedPanel from './NodeLinkedPanel.svelte';
	import MetadataPanel from '$lib/editor/MetadataPanel.svelte';
	import { asStateObject } from '$lib/workspace/panelState';

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
			<select
				class="slot-pick"
				value={curSlot(node) ?? ''}
				onchange={(e) => props.setState({ ...st(), slot: e.currentTarget.value })}
				data-testid="metadata-slot"
			>
				{#each slots as s (s)}
					<option value={s}>{s}</option>
				{/each}
			</select>
		{/if}
	{/snippet}

	{#snippet content(node)}
		<div class="scroll">
			<MetadataPanel {node} showHeader={false} slot={curSlot(node)} />
		</div>
	{/snippet}
</NodeLinkedPanel>

<style>
	.scroll {
		height: 100%;
		overflow-y: auto;
		min-height: 0;
	}
	.slot-pick {
		flex: 0 1 auto;
		min-width: 0;
		font-family: var(--font-mono);
		font-size: 0.78rem;
		padding: 2px 6px;
	}
</style>
