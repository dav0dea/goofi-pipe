<!-- Floating chip counting the currently-errored nodes; click to list them and focus one. -->
<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import { Chip, StatusDot, Popover } from '$lib/ui';

	type Props = { onFocus: (uid: string) => void };
	const { onFocus }: Props = $props();

	const g = graph();
	const activeNodes = $derived(g.nodes.filter((n) => n.error));

	let chipOpen = $state(false);
	let anchorEl = $state<HTMLElement | null>(null);
	function focus(uid: string): void {
		onFocus(uid);
		chipOpen = false;
	}
</script>

{#if activeNodes.length > 0}
	<div class="chip-host" data-testid="error-chip">
		<span class="chip-anchor" bind:this={anchorEl}>
			<Chip tone="danger" aria-expanded={chipOpen} onclick={() => (chipOpen = !chipOpen)}>
				<StatusDot tone="error" size="sm" />
				{activeNodes.length}
				{activeNodes.length === 1 ? 'error' : 'errors'}
			</Chip>
		</span>
		<!-- `flip`: the chip sits at the editor's BOTTOM, so a shifted surface would bury the chip. -->
		<Popover anchor={anchorEl} open={chipOpen} onDismiss={() => (chipOpen = false)} flip>
			<div class="error-list" data-testid="error-popover">
				{#each activeNodes as n (n.uid)}
					<button class="prow" onclick={() => focus(n.uid)}>
						<span class="row-name">{n.name}</span>
						<span class="row-error">{n.error}</span>
					</button>
				{/each}
			</div>
		</Popover>
	</div>
{/if}

<style>
	.chip-host {
		position: absolute;
		left: 12px;
		bottom: 12px;
		z-index: var(--z-chip);
	}
	.chip-anchor {
		display: inline-flex;
	}
	/* The mono is stated HERE, inside the portal: Popover moves this surface to <body>, so it does
	   not inherit the chip's face. */
	.error-list {
		width: 320px;
		max-height: 60dvh;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
		font-family: var(--font-mono);
	}
	.prow {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: var(--space-2);
		background: transparent;
		border: none;
		border-radius: var(--radius-sm);
		padding: var(--space-3) var(--space-5);
		text-align: left;
		cursor: pointer;
		color: var(--text);
		/* The whole font, not just the family: a <button> inherits none of it. */
		font: inherit;
		transition: background var(--dur-fast) var(--ease);
	}
	.prow:hover {
		background: var(--surface-3);
	}
	.row-name {
		color: var(--accent);
		font-size: var(--fs-small);
	}
	.row-error {
		color: var(--text-dim);
		font-size: var(--fs-micro);
		white-space: pre-wrap;
		overflow: hidden;
		text-overflow: ellipsis;
		display: -webkit-box;
		-webkit-line-clamp: 3;
		line-clamp: 3;
		-webkit-box-orient: vertical;
	}
</style>
