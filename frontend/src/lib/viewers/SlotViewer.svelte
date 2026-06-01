<script lang="ts">
	import { subscribeFrames } from '$lib/api/frames';
	import type { DataFrame } from '$lib/codec/decode';
	import { Handle, Position } from '@xyflow/svelte';
	import ViewerSurface from './ViewerSurface.svelte';
	import { viewerKind, cycleViewerKind } from './viewerState.svelte';
	import { ui } from '$lib/stores/ui.svelte';
	import { graph } from '$lib/stores/graph.svelte';
	import { dtypeColor } from '$lib/editor/categoryColor';
	import { onMount } from 'svelte';

	type Props = { node: string; slot: string; dtype: string };
	const { node, slot, dtype }: Props = $props();

	const g = graph();
	const uiStore = ui();

	function onSlotClick(e: MouseEvent): void {
		// Connected slots fall through so SvelteFlow's drag-start handler
		// picks the gesture up; otherwise pop the seeded auto-link menu.
		if (g.isOutputConnected(node, slot)) return;
		e.stopPropagation();
		ui().requestSlotClick({ node, slot, dtype, side: 'source', clientX: e.clientX, clientY: e.clientY });
	}

	// Chosen viewer type lives in a keyed store, so the cycle button, a restored
	// patch, and the agent surface all drive the same place; frame→component
	// dispatch is delegated to <ViewerSurface>.
	const kind = $derived(viewerKind(node, slot, dtype));
	const expanded = $derived(uiStore.isSlotExpanded(node, slot));

	let frame = $state<DataFrame | null>(null);
	let visible = $state(false);
	let container: HTMLDivElement | null = $state(null);

	function toggleExpanded(): void {
		// Bound on the entire header — clicking anywhere except the cycle
		// button or an unconnected slot name (both stopPropagation) toggles
		// the viewer.
		uiStore.toggleSlotExpanded(node, slot);
	}

	function cycle(e: MouseEvent): void {
		// Stop the click from bubbling to the header's collapse toggle.
		e.stopPropagation();
		cycleViewerKind(node, slot, dtype);
	}

	// IntersectionObserver: only subscribe while the viewer is in the
	// viewport. This is crucial for performance with many simultaneous
	// viewers — the data WS opens lazily and tears down when scrolled away.
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
		// Don't burn a WS subscription on a collapsed or off-screen viewer; the
		// frame resets when the target slot changes so a stale view never shows.
		frame = null;
		if (!visible || !expanded) return;
		const unsub = subscribeFrames(node, slot, (f) => (frame = f));
		return () => unsub();
	});
</script>

<div
	class="slot-viewer"
	class:active={visible}
	class:collapsed={!expanded}
	style="--dtype: {dtypeColor(dtype)};"
	bind:this={container}
	data-node={node}
	data-slot={slot}
>
	<header onclick={toggleExpanded} role="button" tabindex="0" aria-expanded={expanded}>
		{#if expanded && dtype === 'ARRAY'}
			<button class="ghost cycle" onclick={cycle} title="cycle viewer">{kind}</button>
		{/if}
		<span
			class="slot-name"
			onclick={onSlotClick}
			role="button"
			tabindex="0"
			data-testid="slot-output"
			title={dtype.toLowerCase()}
		>
			{slot}
		</span>
		<Handle id={slot} type="source" position={Position.Right} />
	</header>

	{#if expanded}
		<div class="body">
			<ViewerSurface {frame} {kind} />
		</div>
	{/if}
</div>

<style>
	.slot-viewer {
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		display: flex;
		flex-direction: column;
		min-height: 80px;
	}
	.slot-viewer.collapsed {
		/* Just the header strip when collapsed — drop the floor on min-height
		   so multiple collapsed viewers stack tight. */
		min-height: 0;
	}
	header {
		display: flex;
		align-items: baseline;
		gap: 6px;
		padding: 2px 6px;
		background: var(--bg-elev-2);
		border-bottom: 1px solid var(--border);
		font-size: 10px;
		cursor: pointer;
		user-select: none;
		transition: background 80ms ease;
		/* Positioning context for the source Handle so SvelteFlow's
		   `top: 50%` anchors to this header row, not the whole node. */
		position: relative;
	}
	header:hover {
		background: var(--bg-elev-3);
	}
	.slot-name {
		/* Hug the right edge of the header so the cycle button (when present)
		   sits flush left and the slot label / handle line up with the node
		   border. */
		margin-left: auto;
		font-family: var(--font-mono);
		/* Slot label colour-codes the dtype: green=array, yellow=string,
		   orange=table. Falls back to dim text for unknown types. */
		color: var(--dtype, var(--text-dim));
		cursor: pointer;
		border-radius: 3px;
		padding: 0 2px;
		transition: background 80ms ease;
	}
	.slot-name:hover {
		background: color-mix(in srgb, var(--dtype, var(--accent)) 18%, transparent);
	}
	/* Inherit the dtype colour for the source handle just like the input
	   handles do over in GoofiNode. */
	header :global(.svelte-flow__handle) {
		background: var(--dtype, var(--bg-elev-3));
		border-color: var(--dtype, var(--border-strong));
	}
	/* Overhang half-outside the node so the pin sits on the outer border.
	   The slot-viewer sits 6px inside the .viewers padding + 1px node
	   border, so push the handle 7px out from the header's right edge —
	   SF's translate(50%, -50%) adds the rest. */
	header :global(.svelte-flow__handle-right) {
		right: -7px;
	}
	.cycle {
		padding: 0 6px;
		font-family: var(--font-mono);
		color: var(--text-dim);
		font-size: 10px;
	}
	.body {
		flex-grow: 1;
		min-height: 0;
		display: flex;
		align-items: stretch;
		justify-content: stretch;
		padding: 4px;
		min-height: 120px;
	}
	.body > :global(*) {
		flex-grow: 1;
		min-width: 0;
		min-height: 0;
	}
</style>
