<!-- Floating error chip — a small, always-visible badge of how many nodes are
     *currently* errored (live snapshot from each node's `error` field, driven by
     the control plane). Click to list them and focus one. Mounted once over the
     editor (AppShell). The full, scrolling log of stdout/stderr + error
     tracebacks lives in the Console panel; this is just the at-a-glance
     indicator, independent of whether a Console is open. -->
<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';

	type Props = { onFocus: (uid: string) => void };
	const { onFocus }: Props = $props();

	const g = graph();
	const activeNodes = $derived(g.nodes.filter((n) => n.error));

	let chipOpen = $state(false);
	function focus(uid: string): void {
		onFocus(uid);
		chipOpen = false;
	}
</script>

{#if activeNodes.length > 0}
	<div class="chip-host" data-testid="error-chip">
		<button class="chip" onclick={() => (chipOpen = !chipOpen)}>
			<span class="dot"></span>
			{activeNodes.length}
			{activeNodes.length === 1 ? 'error' : 'errors'}
		</button>
		{#if chipOpen}
			<div class="popover">
				{#each activeNodes as n (n.uid)}
					<button class="prow" onclick={() => focus(n.uid)}>
						<span class="row-name">{n.name}</span>
						<span class="row-error">{n.error}</span>
					</button>
				{/each}
			</div>
		{/if}
	</div>
{/if}

<style>
	.chip-host {
		position: absolute;
		left: 12px;
		bottom: 12px;
		z-index: var(--z-chip);
		font-family: var(--font-mono);
	}
	.chip {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 4px 10px;
		background: color-mix(in srgb, var(--danger) 18%, var(--bg-elev-1));
		border: 1px solid var(--danger);
		color: var(--text);
		font-size: 11px;
		border-radius: 999px;
		cursor: pointer;
		box-shadow: 0 2px 12px rgba(240, 96, 128, 0.25);
	}
	.chip .dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--danger);
		box-shadow: 0 0 6px var(--danger);
	}
	.popover {
		position: absolute;
		left: 0;
		bottom: calc(100% + 6px);
		width: 320px;
		max-height: 60vh;
		overflow-y: auto;
		background: var(--bg-elev-2);
		border: 1px solid var(--border-strong);
		border-radius: var(--radius-sm);
		box-shadow: var(--shadow-2);
		padding: 4px;
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.prow {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 4px;
		background: transparent;
		border: none;
		border-radius: var(--radius-sm);
		padding: 6px 8px;
		text-align: left;
		cursor: pointer;
		color: var(--text);
		font-family: var(--font-mono);
	}
	.prow:hover {
		background: var(--bg-elev-3);
	}
	.row-name {
		color: var(--accent);
		font-size: 11px;
	}
	.row-error {
		color: var(--text-dim);
		font-size: 10px;
		white-space: pre-wrap;
		overflow: hidden;
		text-overflow: ellipsis;
		display: -webkit-box;
		-webkit-line-clamp: 3;
		line-clamp: 3;
		-webkit-box-orient: vertical;
	}
</style>
