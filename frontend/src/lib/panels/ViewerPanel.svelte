<!-- Viewer panel — visualizes an output slot of the node dragged into it. The
     slot picker plus the shared viewer-type dropdown + settings cog live in the
     panel header (next to the node name); the body is the shared ViewerFeed. The
     viewer type + settings come from the same per-(node, slot) stores the canvas
     SlotViewer uses, so the two views stay in lock-step. Only the chosen slot is
     panel-local state. Linking + empty state via NodeLinkedPanel. -->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import type { NodeInstanceInfo } from '$lib/api/control';
	import NodeLinkedPanel from './NodeLinkedPanel.svelte';
	import ViewerFeed from '$lib/viewers/ViewerFeed.svelte';
	import ViewerControls from '$lib/viewers/ViewerControls.svelte';
	import { asStateObject } from '$lib/workspace/panelState';

	interface ViewerState {
		node?: string | null;
		slot?: string | null;
	}

	let props: PanelProps = $props();

	function st(): ViewerState {
		return asStateObject(props.state) as ViewerState;
	}
	function curSlot(node: NodeInstanceInfo): string | null {
		const cur = st();
		const names = Object.keys(node.output_slots);
		return cur.slot && node.output_slots[cur.slot] ? cur.slot : (names[0] ?? null);
	}
	function pick(slot: string): void {
		props.setState({ ...st(), slot });
	}
</script>

<NodeLinkedPanel {...props} label="data">
	{#snippet controls(node)}
		{@const slot = curSlot(node)}
		{@const dtype = slot ? node.output_slots[slot] : null}
		<select
			class="slot-pick"
			value={slot ?? ''}
			onchange={(e) => pick(e.currentTarget.value)}
			data-testid="viewer-slot"
		>
			{#each Object.entries(node.output_slots) as [name, dt] (name)}
				<option value={name}>{name} · {dt.toLowerCase()}</option>
			{/each}
		</select>
		{#if slot && dtype}
			<ViewerControls node={node.name} {slot} {dtype} />
		{/if}
	{/snippet}

	{#snippet content(node)}
		{@const slot = curSlot(node)}
		{@const dtype = slot ? node.output_slots[slot] : null}
		<div class="vp-body"><ViewerFeed node={node.name} {slot} {dtype} /></div>
	{/snippet}
</NodeLinkedPanel>

<style>
	.slot-pick {
		flex: 0 1 auto;
		min-width: 0;
		font-size: 0.78rem;
		padding: 2px 6px;
	}
	.vp-body {
		flex: 1;
		min-height: 0;
		display: flex;
		padding: 6px;
	}
</style>
