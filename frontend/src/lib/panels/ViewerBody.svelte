<!--
  The viewer body for a linked node: a slot picker + the matching viewer
  component. Subscription is lazy (IntersectionObserver) so an off-screen panel
  costs nothing. The node is provided by ViewerPanel/NodeLinkedPanel; slot/kind
  live in the panel state and are reported back via onPick/onKind.
-->
<script lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { subscribeData } from '$lib/api/data';
	import { isArrayFrame, isStringFrame, isTableFrame, type DataFrame } from '$lib/codec/decode';
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

	let {
		node,
		slotName,
		kind,
		onPick,
		onKind
	}: {
		node: NodeInstanceInfo;
		slotName: string | null;
		kind: ViewerKind;
		onPick: (slot: string) => void;
		onKind: (kind: ViewerKind) => void;
	} = $props();

	const slots = $derived(Object.entries(node.output_slots));
	const dtype = $derived(slotName ? (node.output_slots[slotName] ?? null) : null);

	function cycle(): void {
		if (dtype === 'STRING' || dtype === 'TABLE') return;
		const i = CYCLE_ARRAY.indexOf(kind as (typeof CYCLE_ARRAY)[number]);
		onKind(CYCLE_ARRAY[(i + 1) % CYCLE_ARRAY.length]);
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
		if (!visible || !slotName) return;
		const unsub = subscribeData(node.name, slotName, (f) => (frame = f));
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

<div class="viewer-body">
	<div class="toolbar">
		<select
			class="pick"
			value={slotName ?? ''}
			onchange={(e) => onPick(e.currentTarget.value)}
			data-testid="viewer-slot"
		>
			{#each slots as [name, dt] (name)}
				<option value={name}>{name} · {dt.toLowerCase()}</option>
			{/each}
		</select>
		{#if dtype === 'ARRAY'}
			<button class="ghost kind" onclick={cycle} title="cycle viewer type">{kind}</button>
		{/if}
		{#if dtype}
			<span class="dtype" style="--dtype: {dtypeColor(dtype)}">{dtype.toLowerCase()}</span>
		{/if}
	</div>

	<div class="body" bind:this={container}>
		{#if !slotName}
			<div class="placeholder">node has no output slots</div>
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
	.viewer-body {
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
		max-width: 60%;
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
