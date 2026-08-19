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

	// The stream's identity, held as a DERIVED so that an equal pair is not a change.
	//
	// A host rebuilds the object it destructures these out of on every graph update — the panel's
	// `{@const {slot} = view(node)}` is one — so the props are RE-ASSIGNED, with the same values,
	// whenever anything in the patch moves. Reading them straight into the effect below tied the
	// subscription to that: collapsing one viewer of a slot dropped and re-took every other
	// viewer's subscription in the same tick, and the refcount passing through zero tore down the
	// shared stream underneath them all.
	const streamNode = $derived(node);
	const streamSlot = $derived(slot);

	$effect(() => {
		frame = null;
		const n = streamNode;
		const s = streamSlot;
		if (!visible || !s) return;
		// Subscribe to the slot's single reduced stream (latest decoded frame at
		// display rate). Kind is NOT part of the stream identity, so a kind switch
		// keeps this subscription and only re-negotiates the ViewSpec below.
		const unsub = subscribeFrames(n, s, (f) => (frame = f));
		return () => unsub();
	});

	// Contribute the capacity-derived ViewSpec whenever the kind or the (quantized)
	// pixel budget changes, while subscribed. The bridge merges every viewer's spec
	// for this slot and reduces each frame to their union; the worker re-sends on
	// reconnect.
	$effect(() => {
		if (!visible || !slot || capW === 0 || capH === 0) return;
		const s = slot;
		setViewSpec(node, s, specToken, viewSpecForKind(kind, capW, capH));
		// Drop this contribution when the deps change (kind/size/visibility) or on
		// unmount, so a stale spec from this viewer can't linger in the merge.
		return () => clearViewSpec(node, s, specToken);
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
