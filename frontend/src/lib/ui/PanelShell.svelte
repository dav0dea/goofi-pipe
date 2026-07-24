<!--
  PanelShell — the panel header contract (spec §2.4, audit Q7/#28), generalising the proven
  `NodeLinkedPanel` chrome so ANY panel contributes `title` / `toolbar` / `actions` to exactly ONE
  header row, killing the 4 divergent second-bars. The header IS a `Bar` (the pusher pattern): the
  `title` (a string or a snippet) and the optional `toolbar` group hug the left, the optional
  `actions` group is pushed to the right. The body fills the remaining column and is a flex column,
  so a `ScrollArea` (or any `flex:1; min-height:0` child) the consumer composes inside `children`
  fills and scrolls it — the shell stays unopinionated about whether the body scrolls.

  Header chrome flows through `Bar`'s `var(--bar-*)` hooks; `class` merged, `data-testid` (and any
  other attribute) forwarded via `...rest`. This is the SHELL only — M migrates NodeLinkedPanel's
  consumers onto it.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';
	import Bar from './Bar.svelte';

	let {
		title,
		toolbar,
		actions,
		class: klass = '',
		children,
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		/** The panel name — a plain string, or a snippet for richer header content. */
		title: string | Snippet;
		/** Panel-specific controls next to the title (group tabs / slot picker / viewer selector). */
		toolbar?: Snippet;
		/** Right-pushed actions (settings, unlink, close). */
		actions?: Snippet;
		children?: Snippet;
	} = $props();

	const titleIsString = $derived(typeof title === 'string');
</script>

<div {...rest} class={`ui-panel-shell ${klass}`.trim()}>
	<Bar class="ui-panel-shell-header">
		{#snippet start()}
			<span class="ui-panel-shell-title">
				{#if titleIsString}
					{title}
				{:else}
					{@render (title as Snippet)()}
				{/if}
			</span>
			{#if toolbar}
				<div class="ui-panel-shell-toolbar">{@render toolbar()}</div>
			{/if}
		{/snippet}
		{#snippet end()}
			{#if actions}
				{@render actions()}
			{/if}
		{/snippet}
	</Bar>
	<div class="ui-panel-shell-body">
		{@render children?.()}
	</div>
</div>

<style>
	.ui-panel-shell {
		display: flex;
		flex-direction: column;
		height: 100%;
		min-height: 0;
		min-width: 0;
	}
	/* The title is the primary scan target — bright, weighted, ellipsis'd if long. */
	.ui-panel-shell-title {
		flex: 0 1 auto;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		color: var(--text);
		font-weight: 600;
		font-size: var(--fs-small);
	}
	/* Hosts the panel's own controls; takes the slack and scrolls horizontally if they overflow. */
	.ui-panel-shell-toolbar {
		flex: 1 1 auto;
		min-width: 0;
		display: flex;
		align-items: center;
		gap: var(--space-4);
		overflow-x: auto;
		overflow-y: hidden;
		scrollbar-width: thin;
	}
	/* The body fills the remaining column as a flex column, so a ScrollArea child fills + scrolls it. */
	.ui-panel-shell-body {
		flex: 1;
		min-height: 0;
		min-width: 0;
		display: flex;
		flex-direction: column;
	}
</style>
