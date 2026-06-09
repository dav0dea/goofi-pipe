<!--
  The viewer body for a linked node: just the matching viewer component for the
  chosen slot/kind. The slot picker + viewer selector live in the panel header
  (ViewerPanel). Subscription is lazy (IntersectionObserver) so an off-screen
  panel costs nothing.
-->
<script lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { subscribeFrames } from '$lib/api/frames';
	import type { DataFrame } from '$lib/codec/decode';
	import type { ViewerKind } from '$lib/viewers/kind';
	import ViewerSurface from '$lib/viewers/ViewerSurface.svelte';
	import { viewerSettings } from '$lib/viewers/viewerSettings.svelte';
	import { onMount } from 'svelte';

	let {
		node,
		slotName,
		kind
	}: {
		node: NodeInstanceInfo;
		slotName: string | null;
		kind: ViewerKind;
	} = $props();

	// Read the same per-slot settings the canvas viewer uses, so the panel
	// reflects whatever was set in the slot's cog menu.
	const settings = $derived(slotName ? viewerSettings(node.name, slotName, kind) : {});

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
		const unsub = subscribeFrames(node.name, slotName, (f) => (frame = f));
		return () => unsub();
	});
</script>

<div class="viewer-body" bind:this={container}>
	{#if !slotName}
		<div class="placeholder">node has no output slots</div>
	{:else}
		<ViewerSurface {frame} {kind} {settings} />
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
