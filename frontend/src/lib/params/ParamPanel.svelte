<script lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';
	import type { ParamDescriptor } from '$lib/api/types';
	import { graph } from '$lib/stores/graph.svelte';
	import { categoryColor, formatName } from '$lib/editor/categoryColor';
	import ParamField from './ParamField.svelte';

	type Props = { node: NodeInstanceInfo | null };
	const { node }: Props = $props();

	const g = graph();

	function setValue(group: string, name: string, value: unknown): void {
		if (!node) return;
		void g.updateParam(node.name, group, name, value);
	}
</script>

<section class="panel">
	{#if !node}
		<div class="empty">
			<div class="empty-title">No node selected</div>
			<div class="empty-sub">Click a node to edit its parameters.</div>
		</div>
	{:else}
		<header>
			<span class="dot" style="background: {categoryColor(node.category)};"></span>
			<div class="titles">
				<div class="title">{formatName(node.type)}</div>
				<div class="sub">{node.name} · {node.category}</div>
			</div>
		</header>
		{#if node.doc}
			<p class="docstring">{node.doc}</p>
		{/if}

		{#each Object.entries(node.params) as [groupName, group] (groupName)}
			<details class="group" open={groupName !== 'common'}>
				<summary>{groupName}</summary>
				<div class="rows">
					{#each Object.entries(group) as [paramName, descriptor] (paramName)}
						<ParamField
							{paramName}
							descriptor={descriptor as ParamDescriptor}
							onCommit={(v) => setValue(groupName, paramName, v)}
						/>
					{/each}
				</div>
			</details>
		{/each}
	{/if}
</section>

<style>
	.panel {
		padding: 12px 12px 0;
		display: flex;
		flex-direction: column;
		gap: 10px;
		min-width: 0;
	}
	.empty {
		padding: 36px 12px;
		text-align: center;
		color: var(--text-dim);
	}
	.empty-title {
		font-size: 13px;
		color: var(--text);
		margin-bottom: 4px;
	}
	.empty-sub {
		font-size: 11px;
	}
	header {
		display: flex;
		gap: 10px;
		align-items: center;
	}
	.dot {
		width: 10px;
		height: 10px;
		border-radius: 50%;
	}
	.title {
		font-size: 14px;
		font-weight: 600;
	}
	.sub {
		color: var(--text-faint);
		font-family: var(--font-mono);
		font-size: 10px;
	}
	.docstring {
		font-size: 11px;
		color: var(--text-dim);
		background: var(--bg-elev-2);
		border-radius: var(--radius-sm);
		padding: 6px 8px;
		white-space: pre-wrap;
		margin: 0;
	}
	details.group {
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		background: var(--bg-elev-2);
	}
	summary {
		cursor: pointer;
		padding: 6px 10px;
		font-weight: 600;
		text-transform: capitalize;
		font-size: 12px;
		list-style: none;
		user-select: none;
	}
	summary::-webkit-details-marker {
		display: none;
	}
	summary::before {
		content: '▸';
		display: inline-block;
		margin-right: 6px;
		color: var(--text-faint);
		transition: transform 80ms ease;
	}
	details[open] summary::before {
		transform: rotate(90deg);
	}
	.rows {
		display: flex;
		flex-direction: column;
		gap: 6px;
		padding: 6px 10px 10px;
	}
</style>
