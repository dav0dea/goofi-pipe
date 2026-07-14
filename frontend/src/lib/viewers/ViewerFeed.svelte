<!--
  The viewer body: a lazily-subscribed Data frame fed into the shared
  ViewerSurface. Owns the IntersectionObserver + data subscription and reads the
  kind + settings from its ViewBinding, so neither the in-canvas SlotViewer nor
  the docked ViewerPanel re-implements any of it. Padding is the caller's concern
  (the node body and the panel body space it differently).
-->
<script lang="ts">
	import { subscribeFrames } from '$lib/api/frames';
	import { setViewSpec, clearViewSpec } from '$lib/api/data';
	import { viewSpecForKind } from './capacity';
	import type { DataFrame } from '$lib/codec/decode';
	import ViewerSurface from './ViewerSurface.svelte';
	import type { ViewBinding } from './viewBinding';
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
	// Stable per-instance token so multiple viewers of one slot fold (not evict).
	const specToken =
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

	$effect(() => {
		frame = null;
		if (!visible || !slot) return;
		// Subscribe (latest decoded frame at display rate); a kind change re-runs
		// this effect → re-subscribes.
		const unsub = subscribeFrames(node, slot, kind, (f) => (frame = f));
		return () => unsub();
	});

	// Push the capacity-derived ViewSpec to the node whenever the kind or the
	// (quantized) pixel budget changes, while subscribed. The node reduces each
	// frame to this before sending (Option C); the worker re-sends on reconnect.
	$effect(() => {
		if (!visible || !slot || capW === 0 || capH === 0) return;
		const s = slot;
		const k = kind;
		setViewSpec(node, s, k, specToken, viewSpecForKind(k, capW, capH));
		// Drop this contribution when the deps change (kind/size/visibility) or on
		// unmount, so a stale spec from this viewer can't linger in the fold.
		return () => clearViewSpec(node, s, k, specToken);
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
