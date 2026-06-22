<!--
  Errors panel (backlog #24) — a dockable list of every node currently reporting an
  error, each with its FULL traceback (the floating error chip clamps to 3 lines;
  this panel does not). Click a node to focus it in the active editor.
-->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import { graph } from '$lib/stores/graph.svelte';
	import { workspace } from '$lib/workspace/workspace.svelte';
	import { editorFor } from './editorCommands';

	// eslint-disable-next-line @typescript-eslint/no-unused-vars
	let {}: PanelProps = $props();

	const g = graph();
	const errored = $derived(g.nodes.filter((n) => n.error));

	function focus(name: string): void {
		editorFor(workspace().activePanelId)?.focusNode(name);
	}
</script>

<div class="errors-panel" data-testid="errors-panel">
	{#if errored.length === 0}
		<div class="empty">No errors</div>
	{:else}
		{#each errored as n (n.name)}
			<button class="err" onclick={() => focus(n.name)} title={`Focus ${n.name}`}>
				<div class="ename">{n.name}</div>
				<pre class="etrace">{n.error}</pre>
			</button>
		{/each}
	{/if}
</div>

<style>
	.errors-panel {
		width: 100%;
		height: 100%;
		overflow: auto;
		padding: 4px;
		display: flex;
		flex-direction: column;
		gap: 4px;
		font-family: var(--font-mono);
		font-size: 11px;
	}
	.empty {
		color: var(--text-faint);
		padding: 8px;
	}
	.err {
		display: flex;
		flex-direction: column;
		gap: 4px;
		align-items: stretch;
		text-align: left;
		background: color-mix(in srgb, var(--danger) 8%, transparent);
		border: 1px solid color-mix(in srgb, var(--danger) 35%, transparent);
		border-radius: var(--radius-sm);
		padding: 6px 8px;
		cursor: pointer;
		color: var(--text);
	}
	.err:hover {
		background: color-mix(in srgb, var(--danger) 14%, transparent);
	}
	.ename {
		color: var(--accent);
		font-weight: 600;
	}
	.etrace {
		margin: 0;
		white-space: pre-wrap;
		word-break: break-word;
		color: var(--text-dim);
		font-size: 10px;
		line-height: 1.35;
	}
</style>
