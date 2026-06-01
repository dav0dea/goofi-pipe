<!-- Errors panel — nodes with errors and their tracebacks. Adapter over the
     existing editor/ErrorPanel (inline mode); clicking an entry selects that
     node so the editor focuses it. -->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import ErrorPanel from '$lib/editor/ErrorPanel.svelte';
	import { selection } from '$lib/stores/selection.svelte';

	// eslint-disable-next-line @typescript-eslint/no-unused-vars
	let {}: PanelProps = $props();
	const sel = selection();
</script>

<div class="wrap">
	<ErrorPanel
		mode="inline"
		onFocus={(name) => sel.activeEditorId && sel.selectNodes(sel.activeEditorId, [name])}
	/>
</div>

<style>
	.wrap {
		height: 100%;
		overflow-y: auto;
		min-height: 0;
	}
</style>
