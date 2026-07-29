<!--
  An In/Out boundary node, shown only INSIDE a sub-patch (enter-to-edit view).
  An In node feeds data into the sub-patch (output handle on its right, toward
  the members); an Out node carries data out (input handle on its left). The
  node name is the sub-patch's boundary slot name; the colour is its data type.
-->
<script lang="ts">
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';
	import { dtypeColor } from './categoryColor';
	import { MODE_ATTRS } from '$lib/ui';

	let { data, selected }: NodeProps = $props();
	const dir = $derived(data.dir as 'in' | 'out');
	const name = $derived(data.name as string);
	const dtype = $derived(data.dtype as string);
	// Unwired = added but not yet connected to a member; shown dimmed/dashed and it
	// exposes no port on the collapsed node until wired.
	const wired = $derived(data.wired !== false);
	// Renaming the portal renames the sub-patch's exposed in/out SLOT (the routing key
	// is unchanged, so wires survive). Double-click the label to edit.
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
	<span class="arrow">{dir === 'in' ? '▸' : '▸'}</span>
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
		/* No outline suppression: the app-wide :focus-visible accent ring (app.css) is the one
		   focus convention — a keyboard rename must show it. */
	}
	.dt {
		color: var(--text-muted);
		font-size: 9px;
	}
	/* Touch: `.lbl-edit` (0,1,0) out-specifies app.css's `input` type floor (0,0,1) — so the 11px
	   face force-zooms iOS on focus — while the `min-height: var(--hit)` it does NOT out-specify
	   swells the field to 44px inside a 26px pill (BOUNDARY.height, frozen and mirrored in
	   nodeMetrics.ts for snapping). Both are answered on the pill's terms: 16px type, and the floor
	   released rather than the canvas grown. The pill does gain a few px while a rename is OPEN,
	   which is transient, touch-only, and cheaper than a rename nobody can read. */
	@media (hover: none) and (pointer: coarse) {
		.lbl-edit {
			min-height: 0;
			font-size: 16px;
		}
	}
</style>
