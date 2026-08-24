<!-- Inspector body for a selected sub-patch node, standing in for ParamForm's param groups. -->
<script lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { graph } from '$lib/stores/graph.svelte';
	import { Button, Badge, Icon } from '$lib/ui';

	let { node }: { node: NodeInstanceInfo } = $props();

	const g = graph();

	const memberCount = $derived(node.subpatch?.memberCount ?? 0);

	function expand(): void {
		void g.expandInstance(node.uid).catch((e) => console.warn('expand failed', e));
	}
</script>

<section class="panel" data-testid="subpatch-inspector">
	<div class="rows">
		<div class="members">
			<span class="glyph"><Icon name="group" /></span>
			<Badge tone="accent" class="member-badge">{memberCount} member{memberCount === 1 ? '' : 's'}</Badge>
		</div>
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
	.members {
		display: flex;
		gap: var(--space-6);
		align-items: center;
	}
	.glyph {
		font-size: var(--fs-strong);
		color: var(--accent);
	}
	.rows {
		display: flex;
		flex-direction: column;
		gap: var(--space-6);
		padding: var(--space-6);
	}
</style>
