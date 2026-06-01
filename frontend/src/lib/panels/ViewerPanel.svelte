<!-- Viewer panel — visualizes an output slot of the node dragged into it.
     Linking + empty state via NodeLinkedPanel; slot/kind persist in the panel
     state alongside the linked node. -->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import NodeLinkedPanel from './NodeLinkedPanel.svelte';
	import ViewerBody from './ViewerBody.svelte';

	type ViewerKind = 'line' | 'image' | 'trajectory' | 'topomap' | 'string' | 'table';
	interface ViewerState {
		node?: string | null;
		slot?: string | null;
		kind?: ViewerKind;
	}

	let props: PanelProps = $props();

	function st(): ViewerState {
		return typeof props.state === 'object' && props.state ? (props.state as ViewerState) : {};
	}
	function defaultKind(dt: string | null): ViewerKind {
		if (dt === 'STRING') return 'string';
		if (dt === 'TABLE') return 'table';
		return 'line';
	}
</script>

<NodeLinkedPanel {...props} label="data">
	{#snippet content(node)}
		{@const slotNames = Object.keys(node.output_slots)}
		{@const cur = st()}
		{@const slot = cur.slot && node.output_slots[cur.slot] ? cur.slot : (slotNames[0] ?? null)}
		{@const dtype = slot ? node.output_slots[slot] : null}
		{@const kind = cur.kind ?? defaultKind(dtype)}
		<ViewerBody
			{node}
			slotName={slot}
			{kind}
			onPick={(s) =>
				props.setState({ ...st(), slot: s, kind: defaultKind(node.output_slots[s] ?? null) })}
			onKind={(k) => props.setState({ ...st(), kind: k })}
		/>
	{/snippet}
</NodeLinkedPanel>
