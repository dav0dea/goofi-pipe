<!-- Viewer panel — visualizes an output slot of the bound node. Each panel is an independent
     viewer instance; the data stream is still shared (one WS per node+slot). -->
<script lang="ts">
	import type { PanelProps } from 'panelty';
	import type { NodeInstanceInfo } from '$lib/api/control';
	import NodeLinkedPanel from './NodeLinkedPanel.svelte';
	import ViewerFeed from '$lib/viewers/ViewerFeed.svelte';
	import ViewerControls from '$lib/viewers/ViewerControls.svelte';
	import { panelBinding, type ViewBinding } from '$lib/viewers/viewBinding';
	import { asStateObject } from 'panelty';
	import { workspace } from 'panelty';
	import { Select } from '$lib/ui';

	interface ViewerState {
		node?: string | null;
		slot?: string | null;
	}

	let props: PanelProps = $props();
	const ws = workspace();

	function st(): ViewerState {
		return asStateObject(props.state) as ViewerState;
	}
	function curSlot(node: NodeInstanceInfo): string | null {
		const cur = st();
		const names = Object.keys(node.output_slots);
		return cur.slot && node.output_slots[cur.slot] ? cur.slot : (names[0] ?? null);
	}
	function pick(slot: string): void {
		ws.setPanelSlot(props.panelId, slot);
	}

	// One place, so the controls and content snippets (separate scopes) don't each re-derive it.
	function view(node: NodeInstanceInfo): { slot: string | null; dtype: string | null; binding: ViewBinding } {
		const slot = curSlot(node);
		const dtype = slot ? node.output_slots[slot] : null;
		const binding = panelBinding(
			() => props.state,
			(s, label) => props.setState(s, 'authored', label),
			dtype
		);
		return { slot, dtype, binding };
	}
</script>

<NodeLinkedPanel {...props} label="data">
	{#snippet controls(node)}
		{@const { slot, dtype, binding } = view(node)}
		<Select
			density="chrome"
			value={slot ?? ''}
			onChange={(v) => pick(v)}
			options={Object.keys(node.output_slots)}
			labels={Object.fromEntries(
				Object.entries(node.output_slots).map(([name, dt]) => [
					name,
					`${node.slot_labels?.[name] ?? name} · ${dt.toLowerCase()}`
				])
			)}
			data-testid="viewer-slot"
		/>
		{#if slot && dtype}
			<ViewerControls {dtype} {binding} />
		{/if}
	{/snippet}

	{#snippet content(node)}
		{@const { slot, binding } = view(node)}
		<div class="vp-body"><ViewerFeed node={node.uid} {slot} {binding} /></div>
	{/snippet}
</NodeLinkedPanel>

<style>
	.vp-body {
		flex: 1;
		min-height: 0;
		display: flex;
		padding: var(--space-3);
	}
</style>
