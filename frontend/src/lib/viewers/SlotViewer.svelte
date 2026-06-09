<script lang="ts">
	import { subscribeFrames } from '$lib/api/frames';
	import type { DataFrame } from '$lib/codec/decode';
	import ViewerSurface from './ViewerSurface.svelte';
	import ViewerSettingsMenu from './ViewerSettingsMenu.svelte';
	import { viewerKind, setViewerKind } from './viewerState.svelte';
	import { viewerSettings, rawViewerSettings } from './viewerSettings.svelte';
	import { ARRAY_KINDS, type ViewerKind } from './kind';
	import { ui } from '$lib/stores/ui.svelte';
	import { graph } from '$lib/stores/graph.svelte';
	import { dtypeColor } from '$lib/editor/categoryColor';
	import { onMount } from 'svelte';

	type Props = { node: string; slot: string; dtype: string };
	const { node, slot, dtype }: Props = $props();

	const g = graph();
	const uiStore = ui();

	function onSlotClick(e: MouseEvent): void {
		// Clicking the slot name opens the add-node menu seeded to wire a new node
		// onto this output — outputs fan out, so this never disconnects existing
		// cables. The matching connector pill (in GoofiNode's overlay) does the
		// same; dragging the pill starts a connection instead.
		e.stopPropagation();
		ui().requestSlotClick({ node, slot, dtype, side: 'source', clientX: e.clientX, clientY: e.clientY });
	}

	const kind = $derived(viewerKind(node, slot, dtype));
	const expanded = $derived(uiStore.isSlotExpanded(node, slot));
	const settings = $derived(viewerSettings(node, slot, kind));

	let frame = $state<DataFrame | null>(null);
	let visible = $state(false);
	let container: HTMLDivElement | null = $state(null);

	function toggleExpanded(): void {
		uiStore.toggleSlotExpanded(node, slot);
	}
	function onTriangle(e: MouseEvent): void {
		e.stopPropagation();
		toggleExpanded();
	}
	function onKindChange(e: Event): void {
		e.stopPropagation();
		setViewerKind(node, slot, (e.currentTarget as HTMLSelectElement).value as ViewerKind);
	}

	// IntersectionObserver: only subscribe while the viewer is in the
	// viewport — the data WS opens lazily and tears down when scrolled away.
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
		if (!visible || !expanded) return;
		const unsub = subscribeFrames(node, slot, (f) => (frame = f));
		return () => unsub();
	});

	// Persist view state (collapse / kind / settings) to the backend so it lands
	// in the .gfi on save. Skips the first run so the seeded-from-patch state
	// isn't echoed straight back; pushes (debounced) on every later change.
	let settled = false;
	$effect(() => {
		const dep = [expanded, kind, JSON.stringify(rawViewerSettings(node, slot))];
		void dep;
		if (!settled) {
			settled = true;
			return;
		}
		g.pushNodeViewers(node);
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
		<button class="tri" class:open={expanded} onclick={onTriangle} aria-label="toggle viewer">
			<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M4 2.5 L8.5 6 L4 9.5 Z" /></svg>
		</button>
		{#if expanded && dtype === 'ARRAY'}
			<select class="kind" value={kind} onchange={onKindChange} onclick={(e) => e.stopPropagation()} title="viewer type">
				{#each ARRAY_KINDS as k (k)}<option value={k}>{k}</option>{/each}
			</select>
		{/if}
		<span class="hspace"></span>
		{#if expanded}
			<ViewerSettingsMenu {node} {slot} {kind} />
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
	</header>

	{#if expanded}
		<div class="body">
			<ViewerSurface {frame} {kind} {settings} />
		</div>
	{/if}
</div>

<style>
	.slot-viewer {
		display: flex;
		flex-direction: column;
		/* No background of its own — the header bar and the viewer body each draw
		   their own, so the slot-viewer box never overhangs the node's rounded
		   corners (which is what was clipping the bottom border). */
	}
	header {
		/* A collapsed slot is exactly one unit tall so slots stack on the node's
		   grid. A faint, dark tint of the slot's dtype colour sets the header bar
		   apart from the node body. The horizontal padding keeps the triangle and
		   the slot name clear of the input/output connectors in GoofiNode's
		   overlay. */
		height: var(--node-u);
		box-sizing: border-box;
		border-top: 1px solid var(--border);
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 0 12px 0 12px;
		background: color-mix(in srgb, var(--dtype, var(--text-dim)) 13%, var(--bg));
		font-size: 10px;
		cursor: pointer;
		user-select: none;
		transition: background 80ms ease;
	}
	header:hover {
		background: color-mix(in srgb, var(--dtype, var(--accent)) 20%, var(--bg));
	}

	/* A proper disclosure triangle: an SVG that rotates open/shut, vertically
	   centred, no button chrome. */
	.tri {
		display: inline-grid;
		place-items: center;
		width: 16px;
		height: 16px;
		padding: 0;
		background: none;
		border: 0;
		cursor: pointer;
		flex-shrink: 0;
	}
	.tri svg {
		width: 11px;
		height: 11px;
		fill: var(--text-dim);
		transform: rotate(0deg);
		transition:
			transform 120ms ease,
			fill 80ms ease;
	}
	.tri.open svg {
		transform: rotate(90deg);
	}
	.tri:hover svg {
		fill: var(--text);
	}

	.kind {
		appearance: none;
		font-family: var(--font-mono);
		font-size: 9px;
		line-height: 1;
		text-align: center;
		text-align-last: center;
		color: var(--text-dim);
		background: color-mix(in srgb, var(--bg) 55%, transparent);
		border: 1px solid var(--border);
		border-radius: 3px;
		padding: 2px 4px;
		cursor: pointer;
	}
	.kind:hover {
		color: var(--text);
		border-color: var(--accent);
	}
	.kind:focus {
		outline: none;
		border-color: var(--accent);
	}
	.hspace {
		flex: 1 1 auto;
	}
	.slot-name {
		/* Sits right against the output connector so the label reads as its name. */
		font-family: var(--font-mono);
		color: var(--dtype, var(--text-dim));
		cursor: pointer;
		border-radius: 3px;
		padding: 0 2px;
		transition: background 80ms ease;
	}
	.slot-name:hover {
		background: color-mix(in srgb, var(--dtype, var(--accent)) 22%, transparent);
	}
	.body {
		height: var(--node-viewer);
		box-sizing: border-box;
		/* Keep the plot inside its slot; the node surface clips the outer corners. */
		overflow: hidden;
		display: flex;
		align-items: stretch;
		justify-content: stretch;
		padding: 4px 6px 7px;
		background: var(--bg);
	}
	.body > :global(*) {
		flex-grow: 1;
		min-width: 0;
		min-height: 0;
	}
</style>
