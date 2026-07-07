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

	// The node is keyed EVERYWHERE by its uid (container, links, restart RPC,
	// selection); `name` is display-only. Pass the uid, render the name.
	function focus(uid: string): void {
		editorFor(workspace().activePanelId)?.focusNode(uid);
	}

	function restart(uid: string): void {
		void g.restartNode(uid).catch((e) => console.warn('restart failed', e));
	}
</script>

<div class="errors-panel" data-testid="errors-panel">
	{#if errored.length === 0}
		<div class="empty">No errors</div>
	{:else}
		{#each errored as n (n.uid)}
			<div class="err">
				<div class="ehead">
					<button class="ename" onclick={() => focus(n.uid)} title={`Focus ${n.name}`}>{n.name}</button>
					<button
						class="restart"
						onclick={() => restart(n.uid)}
						title="Restart this node (respawn with the same params + links)"
						data-testid="errors-restart">↻ Restart</button
					>
				</div>
				<pre class="etrace">{n.error}</pre>
			</div>
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
		color: var(--text);
	}
	.ehead {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
	}
	.ename {
		color: var(--accent);
		font-weight: 600;
		background: none;
		border: none;
		padding: 0;
		font: inherit;
		cursor: pointer;
		text-align: left;
	}
	.ename:hover {
		text-decoration: underline;
	}
	.restart {
		flex: 0 0 auto;
		font-size: 10px;
		padding: 2px 7px;
		border-radius: var(--radius-sm);
		border: 1px solid color-mix(in srgb, var(--danger) 45%, transparent);
		background: color-mix(in srgb, var(--danger) 12%, transparent);
		color: var(--text);
		cursor: pointer;
	}
	.restart:hover {
		background: color-mix(in srgb, var(--danger) 22%, transparent);
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
