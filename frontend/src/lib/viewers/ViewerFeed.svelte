<!--
  The viewer body: a lazily-subscribed Data frame fed into the shared
  ViewerSurface. Owns the IntersectionObserver + data subscription and resolves
  the slot's kind + settings from the shared stores, so neither the in-canvas
  SlotViewer nor the docked ViewerPanel re-implements any of it. Padding is the
  caller's concern (the node body and the panel body space it differently).
-->
<script lang="ts">
	import { subscribeFrames } from '$lib/api/frames';
	import type { DataFrame } from '$lib/codec/decode';
	import ViewerSurface from './ViewerSurface.svelte';
	import type { ViewBinding } from './viewBinding';
	import { onMount } from 'svelte';

	let {
		node,
		slot,
		binding
	}: { node: string; slot: string | null; binding: ViewBinding } = $props();

	const kind = $derived(binding.kind);
	const settings = $derived(binding.settings);

	let frame = $state<DataFrame | null>(null);
	let visible = $state(false);
	let container: HTMLDivElement | null = $state(null);

	// IntersectionObserver: only subscribe while the viewer is in the viewport —
	// the data WS opens lazily and tears down when scrolled away.
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
		if (!visible || !slot) return;
		const unsub = subscribeFrames(node, slot, (f) => (frame = f));
		return () => unsub();
	});
</script>

<div class="viewer-feed" bind:this={container}>
	{#if !slot}
		<div class="placeholder">node has no output slots</div>
	{:else}
		<ViewerSurface {frame} {kind} {settings} />
	{/if}
</div>

<style>
	.viewer-feed {
		position: relative;
		flex: 1;
		min-width: 0;
		min-height: 0;
		display: flex;
		align-items: stretch;
		justify-content: stretch;
	}
	.viewer-feed > :global(*) {
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
