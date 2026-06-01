<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';

	type Props = {
		onFocus: (name: string) => void;
		/** `'inline'` renders the full list inside its parent (param panel).
		 * `'chip'` shows a small floating badge anchored to the canvas corner
		 * with a click-to-expand popover. */
		mode?: 'inline' | 'chip';
	};
	const { onFocus, mode = 'inline' }: Props = $props();

	const g = graph();
	const errored = $derived(g.nodes.filter((n) => n.error));

	let chipOpen = $state(false);
	function focus(name: string): void {
		onFocus(name);
		chipOpen = false;
	}
</script>

{#if errored.length > 0 && mode === 'inline'}
	<section class="panel">
		<header>
			<span class="badge">{errored.length}</span>
			<span>Errors</span>
		</header>
		<ul>
			{#each errored as n (n.name)}
				<li>
					<button class="link" onclick={() => focus(n.name)}>{n.name}</button>
					<pre>{n.error}</pre>
				</li>
			{/each}
		</ul>
	</section>
{:else if errored.length > 0 && mode === 'chip'}
	<div class="chip-host" data-testid="error-chip">
		<button class="chip" onclick={() => (chipOpen = !chipOpen)}>
			<span class="dot"></span>
			{errored.length}
			{errored.length === 1 ? 'error' : 'errors'}
		</button>
		{#if chipOpen}
			<div class="popover">
				{#each errored as n (n.name)}
					<button class="row" onclick={() => focus(n.name)}>
						<span class="row-name">{n.name}</span>
						<span class="row-error">{n.error}</span>
					</button>
				{/each}
			</div>
		{/if}
	</div>
{/if}

<style>
	.panel {
		padding: 12px;
		border-top: 1px solid var(--border);
		background: var(--bg-elev-1);
	}
	header {
		display: flex;
		align-items: center;
		gap: 8px;
		font-weight: 600;
		color: var(--danger);
		margin-bottom: 8px;
	}
	.badge {
		background: var(--danger);
		color: #1a0709;
		border-radius: 4px;
		padding: 1px 6px;
		font-size: 10px;
	}
	ul {
		list-style: none;
		padding: 0;
		margin: 0;
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.link {
		background: transparent;
		border: none;
		color: var(--accent);
		padding: 0;
		cursor: pointer;
		font-family: var(--font-mono);
		font-size: 11px;
		text-align: left;
	}
	pre {
		font-family: var(--font-mono);
		font-size: 10px;
		color: var(--text-dim);
		white-space: pre-wrap;
		margin: 4px 0 0;
		max-height: 180px;
		overflow: auto;
	}
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
	.row {
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
	.row:hover {
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
		-webkit-box-orient: vertical;
	}
</style>
