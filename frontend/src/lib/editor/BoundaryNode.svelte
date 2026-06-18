<!--
  An In/Out boundary node, shown only INSIDE a sub-patch (enter-to-edit view).
  An In node feeds data into the sub-patch (output handle on its right, toward
  the members); an Out node carries data out (input handle on its left). The
  node name is the sub-patch's boundary slot name; the colour is its data type.
-->
<script lang="ts">
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';
	import { dtypeColor } from './categoryColor';

	let { data, selected }: NodeProps = $props();
	const dir = $derived(data.dir as 'in' | 'out');
	const name = $derived(data.name as string);
	const dtype = $derived(data.dtype as string);
	// Unwired = added but not yet connected to a member; shown dimmed/dashed and it
	// exposes no port on the collapsed node until wired.
	const wired = $derived(data.wired !== false);
</script>

<div
	class="boundary {dir}"
	class:selected
	class:unwired={!wired}
	style="--dtype: {dtypeColor(dtype)};"
	title="{dir === 'in' ? 'Input' : 'Output'} · {dtype.toLowerCase()}{wired
		? ''
		: ' · unwired (connect to a node)'}"
	data-testid="boundary-node"
>
	<span class="arrow">{dir === 'in' ? '▸' : '▸'}</span>
	<span class="lbl">{name}</span>
	<span class="dt">{dtype.toLowerCase()}</span>
	{#if dir === 'in'}
		<Handle id="out" type="source" position={Position.Right} />
	{:else}
		<Handle id="in" type="target" position={Position.Left} />
	{/if}
</div>

<style>
	.boundary {
		position: relative;
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 5px 10px;
		min-width: 96px;
		font-family: var(--font-mono);
		font-size: 11px;
		color: var(--text);
		background: var(--bg-elev-2);
		border: 1px solid var(--dtype, var(--border-strong));
		border-radius: 999px;
	}
	.boundary.selected {
		box-shadow: 0 0 0 1px var(--dtype, var(--accent));
	}
	.boundary.unwired {
		border-style: dashed;
		opacity: 0.6;
	}
	.boundary.out {
		flex-direction: row-reverse;
	}
	.arrow {
		color: var(--dtype, var(--accent));
	}
	.boundary.out .arrow {
		transform: scaleX(-1);
	}
	.lbl {
		font-weight: 600;
	}
	.dt {
		color: var(--text-faint);
		font-size: 9px;
	}
</style>
