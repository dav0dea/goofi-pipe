<!-- The content a freshly-split panel starts as: a grid of buttons, one per
     registered panel type, that turns this empty panel into the chosen content.
     An APP panel like any other — the panel framework mints the `empty` type and
     draws whatever is registered for it, and what a blank panel offers is a
     question about this app's vocabulary, not about panelling.
     The choices are discovered from the panel registry — the single source of
     truth — so any registered panel (including future mods) shows up here
     automatically, with no list to keep in sync. The grid itself is the shared
     `ChoiceGrid` primitive, which the agent panel's launcher wears too. -->
<script lang="ts">
	import { listPanelTypes, type PanelProps } from 'panelty';
	import { workspace } from 'panelty';
	import { EMPTY_PANEL_TYPE } from '$lib/api/vocab';
	import { ChoiceGrid, EmptyState, type Choice } from '$lib/ui';

	let { panelId }: PanelProps = $props();
	const ws = workspace();

	// Every registered type except 'empty' itself — picking one converts this
	// panel into that content.
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
	/* Stretches so the EmptyState spans the panel width — the grid below the prompt sizes
	   against it (and centres itself at its max-width), rather than shrinking to content. */
	.empty {
		display: flex;
		flex-direction: column;
		justify-content: center;
		height: 100%;
		background: var(--bg);
	}
</style>
