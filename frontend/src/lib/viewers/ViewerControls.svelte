<!--
  Shared viewer header controls: the ARRAY viewer-type dropdown plus the settings
  cog. Used by both the in-canvas SlotViewer header and the docked ViewerPanel
  header. Driven entirely by a ViewBinding (the per-instance view state), so each
  viewer instance picks its own type/settings independently.

  The dropdown is the `Select` primitive at `density="chrome"`. That compact face
  used to be hardcoded here — which is why this one dropdown looked right in a bar
  and every other one looked oversized; it is the primitive's own now, and the
  frozen 24px node-slot header still opts out through `--select-*` (SlotViewer).
-->
<script lang="ts">
	import ViewerSettingsMenu from './ViewerSettingsMenu.svelte';
	import { ARRAY_KINDS, type ViewerKind } from './kind';
	import type { ViewBinding } from './viewBinding';
	import { Select } from '$lib/ui';

	let { dtype, binding }: { dtype: string; binding: ViewBinding } = $props();

	const kind = $derived(binding.kind);
</script>

<!-- stopPropagation so picking a kind on a node header doesn't also toggle the
     slot's collapse; harmless in the panel header. -->
<div class="viewer-controls">
	{#if dtype === 'ARRAY'}
		<Select
			density="chrome"
			value={kind}
			options={[...ARRAY_KINDS]}
			onChange={(k) => binding.setKind(k as ViewerKind)}
			onclick={(e) => e.stopPropagation()}
			title="viewer type"
			data-testid="viewer-kind"
		/>
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
</style>
