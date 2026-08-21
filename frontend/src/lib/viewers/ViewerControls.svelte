<!-- Shared viewer header controls: the ARRAY viewer-type dropdown plus the settings cog. -->
<script lang="ts">
	import ViewerSettingsMenu from './ViewerSettingsMenu.svelte';
	import { ARRAY_KINDS, type ViewerKind } from './kind';
	import type { ViewBinding } from './viewBinding';
	import { Select } from '$lib/ui';

	let { dtype, binding }: { dtype: string; binding: ViewBinding } = $props();

	const kind = $derived(binding.kind);
</script>

<!-- No wrapper: these are two controls in the host strip, which owns their gap. stopPropagation
     so picking a kind on a node header does not also toggle the slot's collapse. -->
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
