<!-- Parameters panel — edits the parameters of the node dragged into it.
     The group tabs live in the panel header bar (next to the node name); the
     body shows just the rows for the active group. Linking + empty state are
     handled by NodeLinkedPanel. -->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import type { NodeInstanceInfo } from '$lib/api/control';
	import NodeLinkedPanel from './NodeLinkedPanel.svelte';
	import ParamPanel, { paramGroupNames } from '$lib/params/ParamPanel.svelte';

	let props: PanelProps = $props();

	function st(): Record<string, unknown> {
		return typeof props.state === 'object' && props.state
			? (props.state as Record<string, unknown>)
			: {};
	}
	// Active group persists in the panel state alongside the linked node; falls
	// back to the first group when unset or stale for the current node.
	function activeGroup(node: NodeInstanceInfo): string | null {
		const groups = paramGroupNames(node);
		const cur = st().group;
		return typeof cur === 'string' && groups.includes(cur) ? cur : (groups[0] ?? null);
	}
</script>

<NodeLinkedPanel {...props} label="parameters">
	{#snippet controls(node)}
		{@const groups = paramGroupNames(node)}
		{@const active = activeGroup(node)}
		<div class="ptabs" role="tablist" data-testid="param-tabs">
			{#each groups as g (g)}
				<button
					class="tab"
					class:active={active === g}
					role="tab"
					aria-selected={active === g}
					onclick={() => props.setState({ ...st(), group: g })}
				>
					{g}
				</button>
			{/each}
		</div>
	{/snippet}

	{#snippet content(node)}
		<div class="scroll">
			<ParamPanel {node} showHeader={false} hideTabs group={activeGroup(node)} />
		</div>
	{/snippet}
</NodeLinkedPanel>

<style>
	.scroll {
		height: 100%;
		overflow-y: auto;
		min-height: 0;
	}
	/* Group tabs, compact for the header bar. */
	.ptabs {
		display: flex;
		align-items: center;
		gap: 2px;
	}
	.tab {
		flex: 0 0 auto;
		padding: 2px 8px;
		background: none;
		border: none;
		border-radius: var(--radius-sm);
		color: var(--text-dim);
		font-family: var(--font-mono);
		font-size: 0.72rem;
		letter-spacing: 0.03em;
		text-transform: lowercase;
		white-space: nowrap;
		cursor: pointer;
		transition:
			color 80ms ease,
			background 80ms ease;
	}
	.tab:hover {
		color: var(--text);
	}
	.tab.active {
		color: var(--text);
		background: color-mix(in srgb, var(--accent) 18%, transparent);
	}
</style>
