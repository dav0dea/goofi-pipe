<!--
  Standalone Viewer panel — point it at any node's output slot for a large,
  dedicated visualization (HD video, EEG, PSD…). Complements the compact
  in-node viewers; reuses the same viewer components and the lazy,
  IntersectionObserver-gated data subscription so off-screen panels cost
  nothing. The chosen node/slot/kind persists in the panel state (and thus the
  saved .gfi layout).
-->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import { subscribeData } from '$lib/api/data';
	import { isArrayFrame, isStringFrame, isTableFrame, type DataFrame } from '$lib/codec/decode';
	import { graph } from '$lib/stores/graph.svelte';
	import { dtypeColor } from '$lib/editor/categoryColor';
	import ArrayViewer from '$lib/viewers/ArrayViewer.svelte';
	import ImageViewer from '$lib/viewers/ImageViewer.svelte';
	import TrajectoryViewer from '$lib/viewers/TrajectoryViewer.svelte';
	import TopomapViewer from '$lib/viewers/TopomapViewer.svelte';
	import StringViewer from '$lib/viewers/StringViewer.svelte';
	import TableViewer from '$lib/viewers/TableViewer.svelte';
	import HighDimFallback from '$lib/viewers/HighDimFallback.svelte';
	import { onMount } from 'svelte';

	const CYCLE_ARRAY = ['line', 'image', 'trajectory', 'topomap'] as const;
	type ViewerKind = (typeof CYCLE_ARRAY)[number] | 'string' | 'table';

	interface ViewerState {
		node: string | null;
		slot: string | null;
		kind: ViewerKind;
	}

	let { state: panelState, setState }: PanelProps = $props();
	const g = graph();

	// Seed once from persisted state; thereafter own the selection locally and
	// write it back on change (no reactive mirror, so no update loop).
	const initial = (panelState ?? {}) as Partial<ViewerState>;
	let selNode = $state<string | null>(initial.node ?? null);
	let selSlot = $state<string | null>(initial.slot ?? null);
	let kind = $state<ViewerKind>(initial.kind ?? 'line');

	function persist(): void {
		setState({ node: selNode, slot: selSlot, kind });
	}

	const nodeOptions = $derived(g.nodes.filter((n) => Object.keys(n.output_slots).length > 0));
	const slotOptions = $derived.by(() => {
		const n = g.nodes.find((nn) => nn.name === selNode);
		return n ? Object.entries(n.output_slots) : [];
	});
	const dtype = $derived.by(() => {
		const n = g.nodes.find((nn) => nn.name === selNode);
		return n && selSlot ? (n.output_slots[selSlot] ?? null) : null;
	});

	function defaultKind(dt: string | null): ViewerKind {
		if (dt === 'STRING') return 'string';
		if (dt === 'TABLE') return 'table';
		return 'line';
	}

	function onNodeChange(name: string): void {
		selNode = name;
		const n = g.nodes.find((nn) => nn.name === name);
		const slots = n ? Object.keys(n.output_slots) : [];
		selSlot = slots[0] ?? null;
		kind = defaultKind(selSlot && n ? n.output_slots[selSlot] : null);
		persist();
	}
	function onSlotChange(slot: string): void {
		selSlot = slot;
		kind = defaultKind(dtype);
		persist();
	}
	function cycleKind(): void {
		if (dtype === 'STRING' || dtype === 'TABLE') return;
		const cur = CYCLE_ARRAY.indexOf(kind as (typeof CYCLE_ARRAY)[number]);
		kind = CYCLE_ARRAY[(cur + 1) % CYCLE_ARRAY.length];
		persist();
	}

	let frame = $state<DataFrame | null>(null);
	let visible = $state(false);
	let container: HTMLDivElement | null = $state(null);

	onMount(() => {
		if (!container) return;
		const obs = new IntersectionObserver(
			(entries) => {
				for (const e of entries) visible = e.isIntersecting;
			},
			{ rootMargin: '64px' }
		);
		obs.observe(container);
		return () => obs.disconnect();
	});

	$effect(() => {
		frame = null;
		if (!visible || !selNode || !selSlot) return;
		const unsub = subscribeData(selNode, selSlot, (f) => (frame = f));
		return () => unsub();
	});

	const arraySpec = $derived.by(() => (frame && isArrayFrame(frame) ? frame.data : null));
	const isRenderable = $derived.by(() => {
		if (!arraySpec) return true;
		const s = arraySpec.shape;
		if (kind === 'line') return s.length <= 3;
		if (kind === 'image') return s.length === 2 || (s.length === 3 && [1, 2, 3, 4].includes(s[2]));
		if (kind === 'trajectory') return s.length === 2 && s[0] >= 2;
		if (kind === 'topomap') return s.length === 1;
		return true;
	});
</script>

<div class="viewer-panel">
	<div class="toolbar">
		<select
			class="pick"
			value={selNode ?? ''}
			onchange={(e) => onNodeChange(e.currentTarget.value)}
			data-testid="viewer-node"
		>
			<option value="" disabled>node…</option>
			{#each nodeOptions as n (n.name)}
				<option value={n.name}>{n.name}</option>
			{/each}
		</select>

		<select
			class="pick"
			value={selSlot ?? ''}
			onchange={(e) => onSlotChange(e.currentTarget.value)}
			disabled={!selNode}
			data-testid="viewer-slot"
		>
			<option value="" disabled>slot…</option>
			{#each slotOptions as [name, dt] (name)}
				<option value={name}>{name} · {dt.toLowerCase()}</option>
			{/each}
		</select>

		{#if dtype === 'ARRAY'}
			<button class="ghost kind" onclick={cycleKind} title="cycle viewer type">{kind}</button>
		{/if}
		{#if dtype}
			<span class="dtype" style="--dtype: {dtypeColor(dtype)}">{dtype.toLowerCase()}</span>
		{/if}
	</div>

	<div class="body" bind:this={container}>
		{#if !selNode || !selSlot}
			<div class="placeholder">Pick a node and output slot to view</div>
		{:else if !frame}
			<div class="placeholder">no data yet</div>
		{:else if !isRenderable && arraySpec}
			<HighDimFallback {arraySpec} />
		{:else if isArrayFrame(frame)}
			{#if kind === 'line'}
				<ArrayViewer {frame} />
			{:else if kind === 'image'}
				<ImageViewer {frame} />
			{:else if kind === 'trajectory'}
				<TrajectoryViewer {frame} />
			{:else if kind === 'topomap'}
				<TopomapViewer {frame} />
			{/if}
		{:else if isStringFrame(frame)}
			<StringViewer {frame} />
		{:else if isTableFrame(frame)}
			<TableViewer {frame} />
		{/if}
	</div>
</div>

<style>
	.viewer-panel {
		display: flex;
		flex-direction: column;
		height: 100%;
		min-height: 0;
	}
	.toolbar {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 5px 8px;
		border-bottom: 1px solid var(--border);
		background: var(--bg-elev-1);
		flex: 0 0 auto;
	}
	.pick {
		font-size: 0.8rem;
		padding: 2px 6px;
		max-width: 40%;
	}
	.kind {
		font-family: var(--font-mono);
		font-size: 0.78rem;
		color: var(--text-dim);
		padding: 2px 8px;
	}
	.dtype {
		margin-left: auto;
		font-size: 0.72rem;
		color: var(--dtype, var(--text-dim));
	}
	.body {
		position: relative;
		flex: 1;
		min-height: 0;
		display: flex;
		padding: 6px;
	}
	.body > :global(*) {
		flex: 1;
		min-width: 0;
		min-height: 0;
	}
	.placeholder {
		flex: 1;
		display: grid;
		place-items: center;
		color: var(--text-faint);
		font-size: 0.82rem;
	}
</style>
