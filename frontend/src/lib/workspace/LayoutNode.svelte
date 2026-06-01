<!--
  Recursive renderer for the layout tree. A `panel` leaf renders `Panel`; a
  `split` lays its children out with flexbox (grow = size fraction) and inserts
  a `Splitter` between each adjacent pair. Children are keyed by id so a panel's
  component instance survives content/state changes and tree reshapes.
-->
<script lang="ts">
	import type { LayoutNode } from './model';
	import { workspace } from './workspace.svelte';
	import Panel from './Panel.svelte';
	import Splitter from './Splitter.svelte';
	import Self from './LayoutNode.svelte';

	let { node }: { node: LayoutNode } = $props();
	const ws = workspace();
</script>

{#if node.kind === 'panel'}
	<Panel {node} />
{:else}
	<div class="split {node.direction}">
		{#each node.children as child, i (child.id)}
			<div class="slot" style="flex-grow: {node.sizes[i]};">
				<Self node={child} />
			</div>
			{#if i < node.children.length - 1}
				<Splitter
					direction={node.direction}
					onResize={(delta) => ws.resize(node.id, i, delta)}
				/>
			{/if}
		{/each}
	</div>
{/if}

<style>
	.split {
		display: flex;
		width: 100%;
		height: 100%;
		min-width: 0;
		min-height: 0;
	}
	.split.row {
		flex-direction: row;
	}
	.split.column {
		flex-direction: column;
	}
	.slot {
		flex-basis: 0;
		min-width: 0;
		min-height: 0;
		display: flex;
	}
</style>
