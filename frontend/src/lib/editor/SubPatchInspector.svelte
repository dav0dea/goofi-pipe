<!--
  Inspector body for a selected sub-patch (virtual) node. Stands in for the param
  groups in ParamPanel. A sub-patch is a purely organizational facade now (no
  sharing), so this shows its name + member count and the one structural action:
  Expand (dissolve back into plain nodes).
-->
<script lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { graph } from '$lib/stores/graph.svelte';

	let { node }: { node: NodeInstanceInfo } = $props();

	const g = graph();

	const instId = $derived(node.subpatch?.instId ?? node.name);
	// Read live instance state so the panel reflects changes immediately (the synth
	// node is rebuilt on the next selection recompute).
	const inst = $derived(g.instances[instId] ?? null);
	const memberCount = $derived(Object.keys(inst?.members ?? {}).length);

	function expand(): void {
		void g.expandInstance(instId).catch((e) => console.warn('expand failed', e));
	}
</script>

<section class="panel" data-testid="subpatch-inspector">
	<header>
		<span class="glyph">▣</span>
		<div class="titles">
			<div class="title">Sub-patch</div>
			<div class="sub">{inst?.name ?? instId}</div>
		</div>
		<span class="badge">{memberCount} member{memberCount === 1 ? '' : 's'}</span>
	</header>

	<div class="rows">
		<button class="act ghost" data-testid="subpatch-expand-inspector" onclick={expand}
			>Expand (dissolve into nodes)</button
		>
	</div>
</section>

<style>
	.panel {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	header {
		display: flex;
		gap: 10px;
		align-items: center;
		padding: 10px 12px;
		border-bottom: 1px solid var(--border);
		background: var(--bg-elev-1);
	}
	.glyph {
		font-size: 14px;
		color: var(--accent);
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
		color: var(--text-dim);
		background: color-mix(in srgb, var(--accent) 14%, transparent);
		border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent);
		flex-shrink: 0;
	}
	.rows {
		display: flex;
		flex-direction: column;
		gap: 12px;
		padding: 12px;
	}
	.act {
		background: transparent;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		cursor: pointer;
		font-size: 11px;
		padding: 5px 9px;
		white-space: nowrap;
	}
	.act:hover {
		border-color: var(--accent);
	}
	.act.ghost {
		color: var(--text-dim);
		justify-content: center;
	}
</style>
