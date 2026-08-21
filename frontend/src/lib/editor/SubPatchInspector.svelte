<!-- Inspector body for a selected sub-patch node, standing in for ParamForm's param groups. -->
<script lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { graph } from '$lib/stores/graph.svelte';
	import { Button, Badge, Icon } from '$lib/ui';

	let { node }: { node: NodeInstanceInfo } = $props();

	const g = graph();

	const instId = $derived(node.subpatch?.instId ?? node.name);
	// Live instance state: the synthetic node is only rebuilt on the next selection recompute.
	const inst = $derived(g.instances[instId] ?? null);
	const memberCount = $derived(Object.keys(inst?.members ?? {}).length);

	function expand(): void {
		void g.expandInstance(instId).catch((e) => console.warn('expand failed', e));
	}
</script>

<section class="panel" data-testid="subpatch-inspector">
	<header>
		<span class="glyph"><Icon name="group" /></span>
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
		gap: var(--space-6);
		align-items: center;
		padding: var(--space-5) var(--space-6);
		border-bottom: 1px solid var(--border);
		background: var(--surface-1);
	}
	.glyph {
		font-size: var(--fs-strong);
		color: var(--accent);
	}
	.titles {
		min-width: 0;
		flex: 1;
	}
	.title {
		font-size: var(--fs-strong);
		font-weight: 600;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.sub {
		color: var(--text-muted);
		font-family: var(--font-mono);
		font-size: var(--fs-micro);
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
		gap: var(--space-6);
		padding: var(--space-6);
	}
</style>
