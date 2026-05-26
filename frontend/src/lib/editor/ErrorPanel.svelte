<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';

	type Props = { onFocus: (name: string) => void };
	const { onFocus }: Props = $props();

	const g = graph();
	const errored = $derived(g.nodes.filter((n) => n.error));
</script>

{#if errored.length > 0}
	<section class="panel">
		<header>
			<span class="badge">{errored.length}</span>
			<span>Errors</span>
		</header>
		<ul>
			{#each errored as n (n.name)}
				<li>
					<button class="link" onclick={() => onFocus(n.name)}>{n.name}</button>
					<pre>{n.error}</pre>
				</li>
			{/each}
		</ul>
	</section>
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
</style>
