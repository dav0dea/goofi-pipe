<script lang="ts">
	import { listPanelTypes, type PanelProps } from 'panelty';
	import { workspace } from 'panelty';
	import { EMPTY_PANEL_TYPE } from '$lib/api/vocab';
	import { ChoiceGrid, EmptyState, type Choice } from '$lib/ui';

	let { panelId }: PanelProps = $props();
	const ws = workspace();

	const choices: Choice[] = listPanelTypes()
		.filter((t) => t.id !== EMPTY_PANEL_TYPE)
		.map((t) => ({
			id: t.id,
			label: t.title,
			icon: t.icon,
			title: t.title,
			choose: () => ws.setType(panelId, t.id)
		}));
</script>

<div class="empty" data-testid="empty-panel">
	<EmptyState>
		{#snippet title()}Choose panel content{/snippet}
		<ChoiceGrid {choices} />
	</EmptyState>
</div>

<style>
	.empty {
		display: flex;
		flex-direction: column;
		justify-content: center;
		height: 100%;
		background: var(--bg);
	}
</style>
