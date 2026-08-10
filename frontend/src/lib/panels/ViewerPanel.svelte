<!-- Viewer panel — visualizes an output slot of the node dragged into it. The
     slot picker plus the shared viewer-type dropdown + settings cog live in the
     panel header (next to the node name); the body is the shared ViewerFeed.
     Each panel is an INDEPENDENT viewer instance: its viewer type + settings
     live in this panel's own layout state (a panelBinding) and persist with the
     panel, so it never tracks the node's inline viewer or another panel. The
     data stream is still shared (one WS per node+slot). Linking + empty state
     via NodeLinkedPanel. -->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import type { NodeInstanceInfo } from '$lib/api/control';
	import NodeLinkedPanel from './NodeLinkedPanel.svelte';
	import ViewerFeed from '$lib/viewers/ViewerFeed.svelte';
	import ViewerControls from '$lib/viewers/ViewerControls.svelte';
	import { panelBinding, type ViewBinding } from '$lib/viewers/viewBinding';
	import { asStateObject } from '$lib/workspace/panelState';
	import { workspace } from '$lib/workspace/workspace.svelte';
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

	// Resolve the chosen slot, its dtype, and this panel's binding in one place so
	// the controls and content snippets (separate scopes) don't each re-derive it.
	// A kind/settings change is a `page_set_panel` command like any other panel write, so the
	// binding's own labelled setter is the whole undo step — no client-side snapshot to capture.
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
