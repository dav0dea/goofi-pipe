<!--
  Shared viewer header controls: the ARRAY viewer-type dropdown plus the settings
  cog. Used by both the in-canvas SlotViewer header and the docked ViewerPanel
  header. Driven entirely by a ViewBinding (the per-instance view state), so each
  viewer instance picks its own type/settings independently. Sizing is
  em-relative so it fits the tiny node header and the larger panel bar alike.
-->
<script lang="ts">
	import ViewerSettingsMenu from './ViewerSettingsMenu.svelte';
	import { ARRAY_KINDS, type ViewerKind } from './kind';
	import type { ViewBinding } from './viewBinding';

	let { dtype, binding }: { dtype: string; binding: ViewBinding } = $props();

	const kind = $derived(binding.kind);

	function onKindChange(e: Event): void {
		// stopPropagation so picking a kind on a node header doesn't also toggle the
		// slot's collapse; harmless in the panel header.
		e.stopPropagation();
		binding.setKind((e.currentTarget as HTMLSelectElement).value as ViewerKind);
	}
</script>

<div class="viewer-controls">
	{#if dtype === 'ARRAY'}
		<select
			class="kind"
			value={kind}
			onchange={onKindChange}
			onclick={(e) => e.stopPropagation()}
			title="viewer type"
		>
			{#each ARRAY_KINDS as k (k)}<option value={k}>{k}</option>{/each}
		</select>
	{/if}
	<ViewerSettingsMenu {binding} />
</div>

<style>
	.viewer-controls {
		display: inline-flex;
		align-items: center;
		gap: var(--space-3);
		flex: 0 0 auto;
	}
	.kind {
		appearance: none;
		font-family: var(--font-mono);
		font-size: var(--fs-small);
		/* The value text rode high at this smaller font-size because the browser
		   floors a <select>'s height (UA min-height), leaving the short line box
		   top-aligned in an over-tall control. Pin the content box to exactly the
		   line height (content-box so they match precisely) and release the UA
		   floor, so the single value line is vertically centered at any font-size. */
		box-sizing: content-box;
		height: 1.5em;
		line-height: 1.5em;
		min-height: 0;
		text-align: center;
		text-align-last: center;
		color: var(--text-dim);
		background: color-mix(in srgb, var(--bg) 55%, transparent);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		padding: 0 var(--space-2);
		cursor: pointer;
	}
	.kind:hover {
		color: var(--text);
		border-color: var(--accent);
	}
	/* Focus chrome is the app-wide base: the global `:focus-visible` accent ring, and nothing else —
	   the old `.kind:focus { outline: none }` only suppressed that ring (0,2,0 beat the 0,1,0 global
	   rule), so it is dropped, not renamed (a literal `:focus-visible { outline: none }` would keep
	   suppressing it). The border stays at rest under focus on purpose: one ring, not two. */

	/* Touch: `.kind` (0,1,0) out-specifies app.css's `select` floor (0,0,1) on BOTH counts — the
	   `min-height: 0` above releases the --hit floor, and --fs-small is 11.5px at the coarse root
	   size, which iOS force-zooms on focus. Restored here.
	   `--viewer-kind-*` is the frozen host's opt-out, the same seam IconButton spends `--icon-btn-
	   size` on: a strip that is shorter than --hit BY CONSTRUCTION states the compact box, and every
	   other host takes the floor. Exactly one host does — SlotViewer's 24px `--node-u` slot header,
	   which is frozen canvas geometry; the docked ViewerPanel header is real chrome and takes it. */
	@media (hover: none) and (pointer: coarse) {
		.kind {
			min-height: var(--viewer-kind-min-h, var(--hit));
			font-size: var(--viewer-kind-fs, 16px);
		}
	}
</style>
