<!--
  One node of the recursive TABLE tree (backlog #13). A leaf renders key → summary;
  a nested TABLE renders an expand/collapse toggle and recurses via <svelte:self>.
  Top level is expanded; deeper levels start collapsed so big tables stay scannable.
-->
<script lang="ts">
	import type { DataFrame } from '$lib/codec/decode';
	import { leafSummary, tableChildren } from './tableTree';
	import { Badge } from '$lib/ui';
	import Self from './TableTree.svelte';

	type Props = { name: string; frame: DataFrame; decimals: number; depth?: number };
	const { name, frame, decimals, depth = 0 }: Props = $props();

	const isTable = $derived(frame.dtype === 'TABLE');
	const children = $derived(tableChildren(frame));
	// null = follow the depth default (top level open); a click pins it explicitly.
	let toggled = $state<boolean | null>(null);
	const expanded = $derived(toggled ?? depth < 1);
</script>

<div class="node" style="padding-left: calc(var(--space-6) * {depth})">
	{#if isTable}
		<button class="row toggle" onclick={() => (toggled = !expanded)} aria-expanded={expanded}>
			<span class="caret">{expanded ? '▾' : '▸'}</span>
			<span class="k">{name}</span>
			<Badge>{children.length}</Badge>
		</button>
		{#if expanded}
			{#each children as [ck, cv] (ck)}
				<Self name={ck} frame={cv} {decimals} depth={depth + 1} />
			{/each}
		{/if}
	{:else}
		<div class="row">
			<span class="k">{name}</span>
			<span class="v" title={leafSummary(frame, decimals)}>{leafSummary(frame, decimals)}</span>
		</div>
	{/if}
</div>

<style>
	.node {
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
	}
	.row {
		display: grid;
		grid-template-columns: auto 110px 1fr;
		gap: var(--space-3);
		align-items: center;
		background: none;
		border: none;
		padding: 0;
		text-align: left;
		font: inherit;
		color: inherit;
	}
	.toggle {
		cursor: pointer;
	}
	.caret {
		color: var(--text-muted);
		width: 10px;
	}
	.k {
		color: var(--text-muted);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.toggle .k {
		grid-column: 2 / 4;
		color: var(--text-dim);
	}
	.v {
		color: var(--text);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
</style>
