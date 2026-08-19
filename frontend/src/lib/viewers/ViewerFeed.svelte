<!--
  The viewer body: a lazily-subscribed Data frame fed into the shared
  ViewerSurface. Owns the IntersectionObserver + data subscription and reads the
  kind + settings from its ViewBinding, so neither the in-canvas SlotViewer nor
  the docked ViewerPanel re-implements any of it. Padding is the caller's concern
  (the node body and the panel body space it differently).
-->
<script lang="ts">
	import { bindViewer } from '$lib/api/frames';
	import { viewSpecForKind } from './capacity';
	import type { DataFrame } from '$lib/codec/decode';
	import ViewerSurface from './ViewerSurface.svelte';
	import type { ViewBinding } from './viewBinding';
	import { EmptyState } from '$lib/ui';
	import { onMount } from 'svelte';

	let { node, slot, binding }: { node: string; slot: string | null; binding: ViewBinding } =
		$props();

	const kind = $derived(binding.kind);
	const settings = $derived(binding.settings);

	let frame = $state<DataFrame | null>(null);
	let visible = $state(false);
	let container: HTMLDivElement | null = $state(null);
	// Device-pixel size of the viewer, quantized to 32-px steps so a 1-px resize
	// doesn't renegotiate the reduction (hysteresis); drives the capacity ViewSpec.
	let capW = $state(0);
	let capH = $state(0);
	// Stable per-instance token so multiple viewers of one slot collect (not evict).
	const token =
		typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `vf-${Math.random()}`;

	function quantize(px: number): number {
		const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
		return Math.max(32, Math.round((px * dpr) / 32) * 32);
	}

	// IntersectionObserver: only subscribe while the viewer is in the viewport —
	// the data WS opens lazily and tears down when scrolled away. ResizeObserver
	// tracks the viewer's pixel budget so the node reduces to what we can show.
	onMount(() => {
		if (!container) return;
		const io = new IntersectionObserver(
			(entries) => {
				for (const e of entries) visible = e.isIntersecting;
			},
			{ rootMargin: '64px' }
		);
		io.observe(container);
		const ro = new ResizeObserver((entries) => {
			for (const e of entries) {
				capW = quantize(e.contentRect.width);
				capH = quantize(e.contentRect.height);
			}
		});
		ro.observe(container);
		return () => {
			io.disconnect();
			ro.disconnect();
		};
	});

	/**
	 * ONE hook for everything about this viewer: which stream it is on, whether it is on screen,
	 * and what it needs the frame reduced to. Binding, unbinding, a resize, a kind switch and a
	 * scroll out of view are the same event — a viewer changed — and they all land in the registry
	 * the same way. What reaches the backend is decided there, from the settled registry, so a
	 * re-run that changes nothing sends nothing, and a re-run while a sibling viewer watches the
	 * same slot cannot disturb it.
	 */
	$effect(() => {
		frame = null;
		if (!visible || !slot) return;
		// Kind is not part of the stream's identity — a kind switch re-negotiates the reduction on
		// the same stream — but it IS part of what this viewer needs, so it belongs in the spec.
		const spec = capW > 0 && capH > 0 ? viewSpecForKind(kind, capW, capH) : null;
		return bindViewer(node, slot, token, spec, (f: DataFrame) => (frame = f));
	});
</script>

<div class="viewer-feed" bind:this={container}>
	{#if !slot}
		<EmptyState>
			{#snippet hint()}node has no output slots{/snippet}
		</EmptyState>
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
</style>
