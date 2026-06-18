<!--
  A collapsed sub-patch, rendered as a single group node. Its handles are the
  instance's boundary interface ports (auto-derived from links crossing the
  group); edges to/from hidden members are rerouted to these handles by the
  editor. Double-click expands (dissolves) the group back into its members.
-->
<script lang="ts">
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';
	import type { InstanceInfo } from '$lib/api/control';

	let { data, selected }: NodeProps = $props();
	const inst = $derived(data.instance as InstanceInfo);
	const instId = $derived(data.instId as string);
	const onExpand = data.onExpand as (id: string) => void;

	const ins = $derived(
		Object.entries(inst?.interface ?? {}).filter(([, p]) => p.dir === 'in')
	);
	const outs = $derived(
		Object.entries(inst?.interface ?? {}).filter(([, p]) => p.dir === 'out')
	);
	const memberCount = $derived(Object.keys(inst?.members ?? {}).length);
	const ROW = 24;
	const HEADER = 30;
	const rows = $derived(Math.max(ins.length, outs.length, 1));
</script>

<div
	class="subpatch-node"
	class:selected
	style="min-height: {HEADER + rows * ROW + 8}px;"
	ondblclick={() => onExpand(instId)}
	role="button"
	tabindex="0"
	title="Double-click to expand this sub-patch"
	data-testid="subpatch-node"
>
	<div class="header">
		<span class="glyph">▣</span>
		<span class="name">{instId}</span>
		<span class="count" title="{memberCount} nodes">{memberCount}</span>
		<button
			class="expand"
			title="Expand sub-patch"
			aria-label="Expand sub-patch"
			data-testid="subpatch-expand"
			onclick={(e) => {
				e.stopPropagation();
				onExpand(instId);
			}}
		>
			⤢
		</button>
	</div>
	<div class="body">sub-patch</div>

	{#each ins as [name], i (name)}
		<div class="conn in" style="top: {HEADER + i * ROW + ROW / 2}px;">
			<Handle id={name} type="target" position={Position.Left} />
			<span class="conn-label left">{name}</span>
		</div>
	{/each}
	{#each outs as [name], i (name)}
		<div class="conn out" style="top: {HEADER + i * ROW + ROW / 2}px;">
			<Handle id={name} type="source" position={Position.Right} />
			<span class="conn-label right">{name}</span>
		</div>
	{/each}
</div>

<style>
	.subpatch-node {
		position: relative;
		width: var(--node-w);
		color: var(--text);
		font-family: var(--font-mono);
		background: var(--bg-elev-2);
		border: 1px solid var(--border-strong);
		border-radius: var(--radius-md);
		box-shadow: var(--shadow-1);
	}
	.subpatch-node.selected {
		border-color: var(--accent);
		box-shadow: 0 0 0 1px var(--accent);
	}
	.header {
		display: flex;
		align-items: center;
		gap: 8px;
		height: 30px;
		padding: 0 10px;
		border-bottom: 1px dashed var(--border);
		background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 22%, transparent), transparent);
		border-top-left-radius: var(--radius-md);
		border-top-right-radius: var(--radius-md);
	}
	.glyph {
		color: var(--accent);
	}
	.name {
		flex: 1 1 auto;
		font-weight: 600;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.count {
		font-size: 10px;
		color: var(--text-faint);
		background: color-mix(in srgb, var(--accent) 16%, transparent);
		border-radius: 8px;
		padding: 1px 7px;
	}
	.expand {
		background: transparent;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-dim);
		cursor: pointer;
		font-size: 11px;
		line-height: 1;
		padding: 2px 5px;
	}
	.expand:hover {
		color: var(--text);
		border-color: var(--accent);
	}
	.body {
		padding: 8px 10px;
		font-size: 10px;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--text-faint);
	}
	.conn {
		position: absolute;
		display: flex;
		align-items: center;
	}
	.conn.in {
		left: -5px;
	}
	.conn.out {
		right: -5px;
		flex-direction: row-reverse;
	}
	.conn-label {
		font-size: 9px;
		color: var(--text-dim);
		background: var(--bg-elev-1);
		border: 1px solid var(--border);
		border-radius: 3px;
		padding: 0 4px;
		max-width: 80px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.conn-label.left {
		margin-left: 8px;
	}
	.conn-label.right {
		margin-right: 8px;
	}
</style>
