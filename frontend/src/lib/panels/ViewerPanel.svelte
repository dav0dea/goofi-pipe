<!-- Viewer panel — visualizes an output slot of the node dragged into it.
     The slot picker + viewer-type selector live in the panel header (next to
     the node name); the body shows the chosen viewer. Linking + empty state via
     NodeLinkedPanel; slot/kind persist in the panel state alongside the node. -->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { dtypeColor } from '$lib/editor/categoryColor';
	import NodeLinkedPanel from './NodeLinkedPanel.svelte';
	import ViewerBody from './ViewerBody.svelte';
	import { asStateObject } from '$lib/workspace/panelState';
	import { defaultKind, cycleKind, type ViewerKind } from '$lib/viewers/kind';

	interface ViewerState {
		node?: string | null;
		slot?: string | null;
		kind?: ViewerKind;
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
	function curKind(node: NodeInstanceInfo, slot: string | null): ViewerKind {
		return st().kind ?? defaultKind(slot ? (node.output_slots[slot] ?? null) : null);
	}
	function pick(node: NodeInstanceInfo, slot: string): void {
		props.setState({ ...st(), slot, kind: defaultKind(node.output_slots[slot] ?? null) });
	}
	function cycle(node: NodeInstanceInfo, slot: string | null, kind: ViewerKind): void {
		if (!slot || node.output_slots[slot] !== 'ARRAY') return;
		props.setState({ ...st(), kind: cycleKind(kind) });
	}
</script>

<NodeLinkedPanel {...props} label="data">
	{#snippet controls(node)}
		{@const slot = curSlot(node)}
		{@const dtype = slot ? node.output_slots[slot] : null}
		{@const kind = curKind(node, slot)}
		<select
			class="slot-pick"
			value={slot ?? ''}
			onchange={(e) => pick(node, e.currentTarget.value)}
			data-testid="viewer-slot"
		>
			{#each Object.entries(node.output_slots) as [name, dt] (name)}
				<option value={name}>{name} · {dt.toLowerCase()}</option>
			{/each}
		</select>
		{#if dtype === 'ARRAY'}
			<button class="kind" onclick={() => cycle(node, slot, kind)} title="cycle viewer type">
				{kind}
			</button>
		{:else if dtype}
			<span class="dtype" style="--dtype: {dtypeColor(dtype)}">{dtype.toLowerCase()}</span>
		{/if}
	{/snippet}

	{#snippet content(node)}
		{@const slot = curSlot(node)}
		{@const kind = curKind(node, slot)}
		<ViewerBody {node} slotName={slot} {kind} />
	{/snippet}
</NodeLinkedPanel>

<style>
	.slot-pick {
		flex: 0 1 auto;
		min-width: 0;
		font-size: 0.78rem;
		padding: 2px 6px;
	}
	.kind {
		flex: 0 0 auto;
		font-family: var(--font-mono);
		font-size: 0.74rem;
		color: var(--text-dim);
		padding: 2px 8px;
	}
	.dtype {
		flex: 0 0 auto;
		font-size: 0.72rem;
		color: var(--dtype, var(--text-dim));
	}
</style>
