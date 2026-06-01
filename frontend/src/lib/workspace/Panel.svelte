<!--
  A leaf of the layout tree: the panel chrome (header) plus the registered
  content component, resolved from the panel type. Clicking anywhere in the
  panel marks it active (capture phase, so it wins even when the click lands on
  inner content) which scopes keyboard shortcuts to the focused panel.
-->
<script lang="ts">
	import type { PanelNode } from './model';
	import { workspace } from './workspace.svelte';
	import { resolvePanelType } from './registry';
	import PanelHeader from './PanelHeader.svelte';

	let { node }: { node: PanelNode } = $props();
	const ws = workspace();
	const type = $derived(resolvePanelType(node.panelType));
	const active = $derived(ws.activePanelId === node.id);
</script>

<section
	class="panel"
	class:active
	onpointerdowncapture={() => ws.setActive(node.id)}
	data-panel-id={node.id}
	data-panel-type={node.panelType}
>
	<PanelHeader {node} />
	<div class="panel-body">
		{#if type.component}
			{@const Content = type.component}
			<Content
				panelId={node.id}
				state={node.state}
				setState={(s) => ws.setPanelState(node.id, s)}
				{active}
			/>
		{:else}
			<div class="missing">Unknown panel type: <code>{node.panelType}</code></div>
		{/if}
	</div>
</section>

<style>
	.panel {
		display: flex;
		flex-direction: column;
		width: 100%;
		height: 100%;
		min-width: 0;
		min-height: 0;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		overflow: hidden;
		/* The active outline sits on the inside so it never shifts layout. */
		box-shadow: inset 0 0 0 1px transparent;
		transition: box-shadow 100ms ease;
	}
	.panel.active {
		box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--accent) 45%, transparent);
	}
	.panel-body {
		position: relative;
		flex: 1;
		min-width: 0;
		min-height: 0;
		overflow: hidden;
	}
	.missing {
		display: grid;
		place-items: center;
		height: 100%;
		color: var(--text-dim);
	}
</style>
