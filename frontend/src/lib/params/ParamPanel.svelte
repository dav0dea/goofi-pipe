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

	// Sort groups: node-specific first (alphabetical), 'common' last.
	const groupEntries = $derived.by(() => {
		if (!node) return [] as [string, Record<string, ParamDescriptor>][];
		const entries = Object.entries(node.params) as [string, Record<string, ParamDescriptor>][];
		return entries.sort(([a], [b]) => {
			if (a === 'common') return 1;
			if (b === 'common') return -1;
			return a.localeCompare(b);
		});
	});
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
			<span class="badge" class:badge-error={Boolean(node.error)} class:badge-ok={!node.error}>
				{node.error ? 'error' : 'running'}
			</span>
		</header>
		{#if node.error}
			<pre class="err" data-testid="param-error">{node.error}</pre>
		{/if}
		{#if node.doc}
			<p class="docstring">{node.doc}</p>
		{/if}

		{#each groupEntries as [groupName, group] (groupName)}
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
		padding: 12px 12px 12px;
		display: flex;
		flex-direction: column;
		gap: 8px;
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
		flex-shrink: 0;
	}
	.titles {
		min-width: 0;
		flex: 1;
	}
	.title {
		font-size: 14px;
		font-weight: 600;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.sub {
		color: var(--text-faint);
		font-family: var(--font-mono);
		font-size: 10px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.badge {
		font-family: var(--font-mono);
		font-size: 9px;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		padding: 2px 6px;
		border-radius: 999px;
		flex-shrink: 0;
	}
	.badge-ok {
		color: var(--success);
		background: color-mix(in srgb, var(--success) 14%, transparent);
		border: 1px solid color-mix(in srgb, var(--success) 30%, transparent);
	}
	.badge-error {
		color: var(--danger);
		background: color-mix(in srgb, var(--danger) 14%, transparent);
		border: 1px solid color-mix(in srgb, var(--danger) 35%, transparent);
	}
	.err {
		margin: 0;
		padding: 8px 10px;
		font-family: var(--font-mono);
		font-size: 10px;
		color: var(--danger);
		background: color-mix(in srgb, var(--danger) 10%, transparent);
		border: 1px solid color-mix(in srgb, var(--danger) 30%, transparent);
		border-radius: var(--radius-sm);
		white-space: pre-wrap;
		word-break: break-word;
		max-height: 140px;
		overflow-y: auto;
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
		gap: 8px;
		padding: 6px 10px 10px;
	}
</style>
