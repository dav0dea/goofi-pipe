<!-- An In/Out boundary node, drawn only INSIDE a sub-patch. -->
<script lang="ts">
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';
	import { BOUNDARY_SLOT } from '$lib/api/vocab';
	import { dtypeColor } from './categoryColor';
	import { Icon, MODE_ATTRS } from '$lib/ui';

	let { data, selected }: NodeProps = $props();
	const dir = $derived(data.dir as 'in' | 'out');
	const name = $derived(data.name as string);
	const dtype = $derived(data.dtype as string);
	const wired = $derived(data.wired !== false);
	// Renames the exposed SLOT only; the routing key is unchanged, so wires survive.
	const rename = $derived(data.rename as ((name: string) => void) | undefined);
	let editing = $state(false);
	let draft = $state('');
	function startEdit() {
		if (!rename) return;
		draft = name;
		editing = true;
	}
	function commit() {
		if (!editing) return;
		editing = false;
		const next = draft.trim();
		if (next && next !== name) rename?.(next);
	}
	function onKey(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			e.preventDefault();
			commit();
		} else if (e.key === 'Escape') {
			e.preventDefault();
			editing = false;
		}
	}
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
	<span class="arrow"><Icon name="chevron-right" /></span>
	{#if editing}
		<!-- svelte-ignore a11y_autofocus -->
		<input
			{...MODE_ATTRS.search}
			class="lbl-edit nodrag"
			bind:value={draft}
			onkeydown={onKey}
			onblur={commit}
			onpointerdown={(e) => e.stopPropagation()}
			autofocus
		/>
	{:else}
		<span class="lbl" ondblclick={startEdit} role="textbox" tabindex="-1" title="Double-click to rename"
			>{name}</span
		>
	{/if}
	<span class="dt">{dtype.toLowerCase()}</span>
	{#if dir === 'in'}
		<Handle id={BOUNDARY_SLOT} type="source" position={Position.Right} />
	{:else}
		<Handle id={BOUNDARY_SLOT} type="target" position={Position.Left} />
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
		background: var(--surface-2);
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
		cursor: text;
	}
	.lbl-edit {
		font-family: var(--font-mono);
		font-size: 11px;
		font-weight: 600;
		color: var(--text);
		background: var(--surface-1);
		border: 1px solid var(--dtype, var(--accent));
		border-radius: 3px;
		padding: 0 3px;
		width: 7ch;
		/* No outline suppression: a keyboard rename must show app.css's :focus-visible ring. */
	}
	.dt {
		color: var(--text-muted);
		font-size: 9px;
	}
	/* Answered on the pill's terms: 16px so iOS does not force-zoom, and the coarse `--hit` floor
	   released rather than the 26px pill (BOUNDARY.height in nodeMetrics.ts) grown to 44px. */
	@media (hover: none) and (pointer: coarse) {
		.lbl-edit {
			min-height: 0;
			font-size: 16px;
		}
	}
</style>
