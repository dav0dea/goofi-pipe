<!-- Floating error chip — a small, always-visible badge of how many nodes are
     *currently* errored (live snapshot from each node's `error` field, driven by
     the control plane). Click to list them and focus one. Mounted once over the
     editor (AppShell). The full, scrolling log of stdout/stderr + error
     tracebacks lives in the Console panel; this is just the at-a-glance
     indicator, independent of whether a Console is open. -->
<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import { Chip, StatusDot, Popover } from '$lib/ui';

	type Props = { onFocus: (uid: string) => void };
	const { onFocus }: Props = $props();

	const g = graph();
	const activeNodes = $derived(g.nodes.filter((n) => n.error));

	let chipOpen = $state(false);
	// The chip is the popover's anchor; wrapping it lets the Popover clamp against a real element
	// and exclude it from its own outside-dismiss (so the toggle click never dismisses-then-reopens).
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
		<!-- `flip`: the chip is anchored to the BOTTOM of the editor, so the plain shift would slide the
		     surface up until it fit and then enclose the chip — burying its own toggle-to-close and the
		     live error count. Opening upward is what this panel did before it delegated to Popover. -->
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
		font-family: var(--font-mono);
	}
	.chip-anchor {
		display: inline-flex;
	}
	/* The popover surface + dismissal come from the Popover primitive; this just sizes and
	   scrolls the error list inside it. */
	.error-list {
		width: 320px;
		max-height: 60vh;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
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
		/* The whole font, not just the family: a <button> inherits none, so the size and
		   line-height would drop to the UA default were app.css's base reset not there. */
		font: inherit;
		/* Likewise the fade on the hover fill below — it came from the base skin M-Task 7 stripped. */
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
