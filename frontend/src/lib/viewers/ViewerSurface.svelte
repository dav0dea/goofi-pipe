<!--
  Renders a decoded Data frame with the chosen viewer kind, falling back to a
  numeric summary when the array shape can't be drawn. This is the single owner
  of the frame→component dispatch shared by the in-canvas SlotViewer and the
  standalone Viewer panel — neither re-implements the if/else chain or the
  renderability check.

  Subscription is the caller's concern (it differs: the canvas viewer also gates
  on collapse). The caller passes the latest `frame` and the resolved `kind`.
-->
<script lang="ts">
	import { isArrayFrame, isStringFrame, isTableFrame, type DataFrame } from '$lib/codec/decode';
	import { isRenderable, type ViewerKind } from './kind';
	import { summaryOf } from './viewMeta';
	import type { SettingsMap } from './settingsSchema';
	import ArrayViewer from './ArrayViewer.svelte';
	import ImageViewer from './ImageViewer.svelte';
	import TrajectoryViewer from './TrajectoryViewer.svelte';
	import TopomapViewer from './TopomapViewer.svelte';
	import StringViewer from './StringViewer.svelte';
	import TableViewer from './TableViewer.svelte';
	import HighDimFallback from './HighDimFallback.svelte';

	let {
		frame,
		kind,
		settings = {}
	}: { frame: DataFrame | null; kind: ViewerKind; settings?: SettingsMap } = $props();

	const arraySpec = $derived(frame && isArrayFrame(frame) ? frame.data : null);
	const renderable = $derived(isRenderable(kind, arraySpec));
	// A locally-non-renderable array (a shape this kind can't draw) resolves to the
	// text fallback — float stats computed from the received frame (Option C frames
	// are float-accurate, so there's no separate backend summary anymore).
	const summary = $derived.by(() => {
		if (!frame || !arraySpec) return null;
		if (!renderable) return summaryOf(arraySpec);
		return null;
	});
</script>

{#if !frame}
	<div class="vs-placeholder">no data yet</div>
{:else if summary}
	<HighDimFallback {summary} />
{:else if isArrayFrame(frame)}
	{#if kind === 'line'}
		<ArrayViewer {frame} {settings} />
	{:else if kind === 'image'}
		<ImageViewer {frame} {settings} />
	{:else if kind === 'trajectory'}
		<TrajectoryViewer {frame} {settings} />
	{:else if kind === 'topomap'}
		<TopomapViewer {frame} {settings} />
	{/if}
{:else if isStringFrame(frame)}
	<StringViewer {frame} {settings} />
{:else if isTableFrame(frame)}
	<TableViewer {frame} {settings} />
{/if}

<style>
	.vs-placeholder {
		flex: 1;
		display: grid;
		place-items: center;
		color: var(--text-faint);
		font-size: 0.82rem;
	}
</style>
