<!--
  Inspector body for a selected sub-patch (virtual) node. Stands in for the param
  groups in ParamForm. A sub-patch is a purely organizational facade now (no
  sharing), so this shows its name + member count and the one structural action:
  Expand (dissolve back into plain nodes).
-->
<script lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { graph } from '$lib/stores/graph.svelte';
	import { Button, Badge } from '$lib/ui';

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
		<Badge tone="accent" class="member-badge">{memberCount} member{memberCount === 1 ? '' : 's'}</Badge>
	</header>

	<div class="rows">
		<Button variant="ghost" data-testid="subpatch-expand-inspector" onclick={expand}
			>Expand (dissolve into nodes)</Button
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
		background: var(--surface-1);
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
		color: var(--text-muted);
		font-family: var(--font-mono);
		font-size: 10px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	header :global(.member-badge) {
		flex-shrink: 0;
	}
	.rows {
		display: flex;
		flex-direction: column;
		gap: 12px;
		padding: 12px;
	}
</style>
