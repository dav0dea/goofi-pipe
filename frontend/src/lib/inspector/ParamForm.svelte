<script module lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';

	/** Parameter group names in display order: node-specific groups alphabetical, 'common' last. */
	export function paramGroupNames(node: NodeInstanceInfo | null): string[] {
		if (!node) return [];
		return Object.keys(node.params).sort((a, b) => {
			if (a === 'common') return 1;
			if (b === 'common') return -1;
			return a.localeCompare(b);
		});
	}
</script>

<!--
  ParamForm — the node-driven inspector: an identity header, the group tab strip it owns, and one
  `<ParamField>` per param in the active group.
-->
<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import type { ParamDescriptor } from '$lib/api/types';
	import { graph } from '$lib/stores/graph.svelte';
	import { formatName } from '$lib/editor/categoryColor';
	import ParamField from './ParamField.svelte';
	import SubPatchInspector from '$lib/editor/SubPatchInspector.svelte';
	import { Bar, Tabs, Badge, Disclosure, EmptyState, Icon, IconButton, MODE_ATTRS } from '$lib/ui';

	let {
		node,
		showHeader = true,
		onClose,
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLElement> & {
		node: NodeInstanceInfo | null;
		/** Show the identity header (rename + state + docs). */
		showHeader?: boolean;
		/** Renders a ✕ in the identity Bar; only the slide-in inspector supplies one. */
		onClose?: () => void;
	} = $props();

	const g = graph();

	// Each RPC is fire-and-forget with a logged failure, so a rejection is never unhandled.
	function setValue(group: string, name: string, value: unknown): void {
		if (!node) return;
		void g.updateParam(node.uid, group, name, value).catch((e) => console.warn('update failed', e));
	}
	function refreshOptions(group: string, name: string): void {
		if (!node) return;
		void g.refreshParam(node.uid, group, name).catch((e) => console.warn('refresh failed', e));
	}
	function setExpression(
		group: string,
		name: string,
		expression: string | null,
		opts: { enabled?: boolean; triggers_process?: boolean } = {}
	): void {
		if (!node) return;
		void g.setExpression(node.uid, group, name, expression, opts).catch((e) =>
			console.warn('set expression failed', e)
		);
	}

	// Keyed by uid, so switching nodes closes the editor while a live state update (which re-creates the
	// node object) leaves an open edit untouched.
	let editingUid = $state<string | null>(null);
	let nameDraft = $state('');
	const editingName = $derived(node != null && editingUid === node.uid);

	function startRename(): void {
		if (!node) return;
		nameDraft = node.name;
		editingUid = node.uid;
	}
	function commitRename(): void {
		// Escape/cancel nulls editingUid first, so the blur the unmounting input fires is a no-op here.
		const uid = editingUid;
		editingUid = null;
		if (!uid || !node || node.uid !== uid) return;
		const base = nameDraft.trim();
		if (!base) return; // empty → keep the current name
		void g.renameNode(uid, base).catch((e) => console.warn('rename failed', e));
	}
	function cancelRename(): void {
		editingUid = null;
	}
	function focusInput(el: HTMLInputElement): void {
		el.focus();
		el.select();
	}

	const groupNames = $derived(paramGroupNames(node));

	const tabItems = $derived(groupNames.map((name) => ({ id: name, label: name })));

	// DERIVED, so the right tab is in the first paint and there is no `.ui-tab` background transition to
	// animate; the effect then ADOPTS the fallback, which keeps the front group sticky.
	let frontGroup = $state<string | null>(null);
	const activeGroup = $derived.by<string | null>(() => {
		const valid = groupNames;
		if (valid.length === 0) return null;
		return frontGroup && valid.includes(frontGroup) ? frontGroup : valid[0];
	});
	$effect(() => {
		if (activeGroup !== frontGroup) frontGroup = activeGroup;
	});

	const activeParams = $derived.by<[string, ParamDescriptor][]>(() => {
		const n = node;
		if (!n || !activeGroup) return [];
		return Object.entries(n.params[activeGroup] ?? {}) as [string, ParamDescriptor][];
	});
</script>

<section {...rest} class={`param-form ${klass}`.trim()}>
	{#if !node}
		<EmptyState data-testid="param-empty">
			{#snippet title()}No node selected{/snippet}
			{#snippet hint()}Select a node to edit its parameters.{/snippet}
		</EmptyState>
	{:else if node.subpatch}
		<SubPatchInspector {node} />
	{:else}
		{#if showHeader}
			<Bar class="pf-identity-bar">
				{#snippet start()}
					<div class="pf-identity">
						<div class="pf-title">
							{#if editingName}
								<!-- svelte-ignore a11y_autofocus -->
								<input
									{...MODE_ATTRS.search}
									class="pf-rename"
									aria-label="Node name"
									value={nameDraft}
									oninput={(e) => (nameDraft = e.currentTarget.value)}
									onblur={commitRename}
									onkeydown={(e) => {
										if (e.key === 'Enter') commitRename();
										else if (e.key === 'Escape') cancelRename();
									}}
									data-testid="node-name-input"
									use:focusInput
								/>
							{:else}
								<button
									class="pf-name"
									title="Click to rename"
									onclick={startRename}
									data-testid="node-name">{node.name}</button
								>
							{/if}
						</div>
						<div class="pf-type">{formatName(node.type)}</div>
					</div>
				{/snippet}
				{#snippet end()}
					<Badge tone={node.error ? 'danger' : 'success'} class="pf-state" data-testid="node-state">
						{node.error ? 'error' : 'running'}
					</Badge>
					{#if onClose}
						<IconButton
							variant="ghost"
							density="chrome"
							class="pf-close"
							label="Close inspector"
							title="Close the inspector"
							data-testid="inspector-close"
							onclick={onClose}><Icon name="x" /></IconButton
						>
					{/if}
				{/snippet}
			</Bar>

			{#if node.doc}
				<Disclosure>
					{#snippet summary()}
						<span data-testid="docs-toggle">docs</span>
					{/snippet}
					{#snippet children()}
						<p class="pf-docstring" data-testid="docstring">{node.doc}</p>
					{/snippet}
				</Disclosure>
			{/if}
		{/if}

		{#if tabItems.length > 0}
			<Tabs
				items={tabItems}
				active={activeGroup ?? undefined}
				onSelect={(id) => (frontGroup = id)}
				data-testid="param-tabs"
			/>
		{/if}

		<!-- A tabpanel only when a tablist exists: an orphaned `tabpanel` role would have no owning tablist. -->
		<div
			class="pf-rows"
			role={tabItems.length > 0 ? 'tabpanel' : undefined}
			aria-label={activeGroup ?? undefined}
			data-testid="param-rows"
		>
			{#if activeParams.length === 0}
				<div class="pf-empty-group" data-testid="param-empty-group">No parameters in this group.</div>
			{:else}
				{#each activeParams as [paramName, descriptor] (node.uid + '/' + paramName)}
					<ParamField
						{paramName}
						{descriptor}
						data-testid={`param-field-${paramName}`}
						refreshing={node != null && g.isRefreshing(node.uid, activeGroup ?? '', paramName)}
						onCommit={(v) => setValue(activeGroup ?? '', paramName, v)}
						onSetExpression={(expr, opts) => setExpression(activeGroup ?? '', paramName, expr, opts)}
						onRefresh={() => refreshOptions(activeGroup ?? '', paramName)}
					/>
				{/each}
			{/if}
		</div>
	{/if}
</section>

<style>
	.param-form {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	.pf-identity {
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
		min-width: 0;
	}
	.pf-title {
		font-size: var(--fs-strong);
		font-weight: 600;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	/* Mono, stated after the `font: inherit` reset that would otherwise wipe it: this is the same
	   identifier the canvas paints on the node. Same for the rename input it swaps with. */
	.pf-name {
		font: inherit;
		font-family: var(--font-mono);
		color: var(--text);
		background: none;
		border: none;
		padding: 0;
		cursor: text;
		border-radius: var(--radius-sm);
		/* The truncation is the BUTTON's own: `text-overflow` on `.pf-title` reaches the text in it,
		   never an overflowing child element, so a long name was cut mid-word with no ellipsis. */
		display: block;
		max-width: 100%;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.pf-name:hover {
		text-decoration: underline;
		text-decoration-style: dotted;
		text-underline-offset: 2px;
	}
	.pf-rename {
		width: 100%;
		font: inherit;
		font-family: var(--font-mono);
		font-size: var(--fs-strong);
		font-weight: 600;
		padding: var(--space-1) var(--space-2);
		color: var(--text);
		background: var(--surface-2);
		border: 1px solid var(--accent);
		border-radius: var(--radius-sm);
	}
	.pf-type {
		color: var(--text-muted);
		font-family: var(--font-mono);
		font-size: var(--fs-micro);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.pf-docstring {
		margin: 0;
		font-size: var(--fs-small);
		color: var(--text-dim);
		white-space: pre-wrap;
	}
	/* Anchored on `.param-form`, a real element of THIS template: `pf-identity-bar` is a class passed to
	   another component, and Svelte's scoping hash never reaches its markup. */
	.param-form :global(.pf-identity-bar) {
		/* The ✕ must never be squeezed into overflow past the pane's edge; the name is what ellipsizes. */
		--bar-end-min: max-content;
		/* Two lines tall by construction, so it takes back the padding a one-row strip has none of. */
		--bar-pad-y: var(--space-2);
	}
	.param-form :global(.pf-identity-bar .pf-close) {
		--panelty-icon-btn-size: 22px;
		color: var(--text-dim);
	}
	.param-form :global(.pf-identity-bar .pf-close:hover) {
		color: var(--text);
	}
	/* Below this the row cannot seat name + state + ✕, and the overflow would walk the ✕ off screen; the
	   badge is what yields. Asked of the PANE, not the host panel. */
	@container (max-width: 180px) {
		.param-form :global(.pf-identity-bar .pf-state) {
			display: none;
		}
	}
	/* The SAME surface the active tab drops to, so the tab merges into the body with no seam line. */
	.pf-rows {
		display: flex;
		flex-direction: column;
		gap: var(--space-5);
		padding: var(--space-6);
		background: var(--surface-1);
	}
	.pf-empty-group {
		color: var(--text-muted);
		font-size: var(--fs-small);
		text-align: center;
		padding: var(--space-6) 0;
	}
	/* The editable cue is a hover underline, so with no hover it rests visible instead. */
	@media (hover: none) and (pointer: coarse) {
		.pf-name {
			text-decoration: underline;
			text-decoration-style: dotted;
			text-underline-offset: 2px;
		}
	}
	/* 16px so focusing it does not force-zoom iOS; mirrors app.css's coarse input floor. */
	@media (hover: none) and (pointer: coarse) {
		.pf-rename {
			font-size: 16px;
		}
	}
</style>
