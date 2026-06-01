<!--
  The viewer body for a linked node: just the matching viewer component for the
  chosen slot/kind. The slot picker + viewer selector live in the panel header
  (ViewerPanel). Subscription is lazy (IntersectionObserver) so an off-screen
  panel costs nothing.
-->
<script lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { subscribeData } from '$lib/api/data';
	import { isArrayFrame, isStringFrame, isTableFrame, type DataFrame } from '$lib/codec/decode';
	import ArrayViewer from '$lib/viewers/ArrayViewer.svelte';
	import ImageViewer from '$lib/viewers/ImageViewer.svelte';
	import TrajectoryViewer from '$lib/viewers/TrajectoryViewer.svelte';
	import TopomapViewer from '$lib/viewers/TopomapViewer.svelte';
	import StringViewer from '$lib/viewers/StringViewer.svelte';
	import TableViewer from '$lib/viewers/TableViewer.svelte';
	import HighDimFallback from '$lib/viewers/HighDimFallback.svelte';
	import { onMount } from 'svelte';

	type ViewerKind = 'line' | 'image' | 'trajectory' | 'topomap' | 'string' | 'table';

	let {
		node,
		slotName,
		kind
	}: {
		node: NodeInstanceInfo;
		slotName: string | null;
		kind: ViewerKind;
	} = $props();

	const dtype = $derived(slotName ? (node.output_slots[slotName] ?? null) : null);

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

<div class="viewer-body" bind:this={container}>
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

<style>
	.viewer-body {
		position: relative;
		flex: 1;
		min-height: 0;
		display: flex;
		padding: 6px;
	}
	.viewer-body > :global(*) {
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
