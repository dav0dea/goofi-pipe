<script lang="ts">
	import { subscribeData } from '$lib/api/data';
	import { isArrayFrame, isStringFrame, isTableFrame, type DataFrame } from '$lib/codec/decode';
	import ArrayViewer from './ArrayViewer.svelte';
	import ImageViewer from './ImageViewer.svelte';
	import TrajectoryViewer from './TrajectoryViewer.svelte';
	import TopomapViewer from './TopomapViewer.svelte';
	import StringViewer from './StringViewer.svelte';
	import TableViewer from './TableViewer.svelte';
	import HighDimFallback from './HighDimFallback.svelte';
	import { ui } from '$lib/stores/ui.svelte';
	import { onMount, untrack } from 'svelte';

	type Props = { node: string; slot: string; dtype: string };
	const { node, slot, dtype }: Props = $props();

	const CYCLE_ARRAY = ['line', 'image', 'trajectory', 'topomap'] as const;
	type ViewerKind = (typeof CYCLE_ARRAY)[number] | 'string' | 'table' | 'fallback';

	function initialKind(dt: string): ViewerKind {
		if (dt === 'STRING') return 'string';
		if (dt === 'TABLE') return 'table';
		return 'line';
	}

	let kind = $state<ViewerKind>(initialKind(dtype));
	let frame = $state<DataFrame | null>(null);
	let visible = $state(false);
	let container: HTMLDivElement | null = $state(null);
	const uiStore = ui();
	const expanded = $derived(uiStore.isSlotExpanded(node, slot));

	function toggleExpanded(e: MouseEvent): void {
		e.stopPropagation();
		uiStore.toggleSlotExpanded(node, slot);
	}

	$effect(() => {
		// Reset frame when target node/slot changes (and kind back to default).
		untrack(() => {
			kind = initialKind(dtype);
		});
		frame = null;
	});

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
		// Don't burn a WS subscription on a collapsed viewer — and tear the
		// existing one down when the user collapses it.
		if (!visible || !expanded) return;
		const unsub = subscribeData(node, slot, (f) => (frame = f));
		return () => unsub();
	});

	function cycle(): void {
		if (dtype === 'STRING' || dtype === 'TABLE') return;
		const cur = CYCLE_ARRAY.indexOf(kind as (typeof CYCLE_ARRAY)[number]);
		const next = CYCLE_ARRAY[(cur + 1) % CYCLE_ARRAY.length];
		kind = next;
	}

	// Detect "we can't draw this" → fall back to a numeric summary.
	const arraySpec = $derived.by(() => {
		if (frame && isArrayFrame(frame)) {
			return frame.data;
		}
		return null;
	});

	const isRenderable = $derived.by(() => {
		if (!arraySpec) return true;
		const s = arraySpec.shape;
		if (kind === 'line') return s.length === 0 || s.length === 1 || s.length === 2 || s.length === 3 ? true : false;
		if (kind === 'image') return (s.length === 2) || (s.length === 3 && [1, 2, 3, 4].includes(s[2]));
		if (kind === 'trajectory') return s.length === 2 && s[0] >= 2;
		if (kind === 'topomap') return s.length === 1;
		return true;
	});
</script>

<div
	class="slot-viewer"
	class:active={visible}
	class:collapsed={!expanded}
	bind:this={container}
	data-node={node}
	data-slot={slot}
>
	<header>
		<span class="slot-name">{slot}</span>
		{#if dtype === 'ARRAY'}
			<button class="ghost cycle" onclick={cycle} title="cycle viewer">{kind}</button>
		{:else}
			<span class="kind">{dtype.toLowerCase()}</span>
		{/if}
		<button
			class="ghost expand"
			onclick={toggleExpanded}
			aria-label={expanded ? 'collapse viewer' : 'expand viewer'}
			title={expanded ? 'collapse' : 'expand'}
		>
			{expanded ? '▾' : '▸'}
		</button>
	</header>

	{#if expanded}
		<div class="body">
			{#if !frame}
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
		overflow: hidden;
	}
	.slot-viewer.collapsed {
		/* Just the header strip when collapsed — drop the floor on min-height
		   so multiple collapsed viewers stack tight. */
		min-height: 0;
	}
	header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 2px 6px;
		background: var(--bg-elev-2);
		border-bottom: 1px solid var(--border);
		font-size: 10px;
	}
	.slot-name {
		font-family: var(--font-mono);
		color: var(--text-dim);
	}
	.kind {
		font-family: var(--font-mono);
		color: var(--text-faint);
	}
	.cycle {
		padding: 0 6px;
		font-family: var(--font-mono);
		color: var(--text-dim);
		font-size: 10px;
		margin-left: auto;
	}
	.expand {
		padding: 0;
		width: 18px;
		height: 18px;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 13px;
		line-height: 1;
		padding-top: 1px;
		color: var(--text-dim);
		border-radius: 50%;
		transition:
			color 80ms ease,
			box-shadow 120ms ease;
	}
	.expand:hover {
		color: var(--accent);
		background: transparent;
		box-shadow:
			0 0 0 1px color-mix(in srgb, var(--accent) 40%, transparent),
			0 0 10px color-mix(in srgb, var(--accent) 50%, transparent);
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
	.placeholder {
		flex-grow: 1;
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--text-faint);
		font-size: 10px;
	}
</style>
