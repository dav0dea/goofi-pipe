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
		gap: 6px;
		flex: 0 0 auto;
	}
	.kind {
		appearance: none;
		font-family: var(--font-mono);
		font-size: 0.85em;
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
</style>
