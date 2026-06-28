<script module lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';

	/** Parameter group names in display order: node-specific groups alphabetical,
	 * 'common' last. Shared with the dedicated Parameters panel, which renders
	 * the group tabs in its own header bar. */
	export function paramGroupNames(node: NodeInstanceInfo | null): string[] {
		if (!node) return [];
		return Object.keys(node.params).sort((a, b) => {
			if (a === 'common') return 1;
			if (b === 'common') return -1;
			return a.localeCompare(b);
		});
	}
</script>

<script lang="ts">
	import type { ParamDescriptor } from '$lib/api/types';
	import { graph } from '$lib/stores/graph.svelte';
	import { categoryColor, formatName } from '$lib/editor/categoryColor';
	import ParamField from './ParamField.svelte';
	import SubPatchInspector from '$lib/editor/SubPatchInspector.svelte';

	type Props = {
		node: NodeInstanceInfo | null;
		/** Show the node identity header (class + status light + name + docs
		 * toggle). True in the editor's slide-in inspector; false in the
		 * dedicated Parameters panel, which already names the node in its linkbar
		 * and wants the group tabs at the very top. */
		showHeader?: boolean;
		/** Hide the internal group-tab strip — the dedicated panel renders the
		 * tabs in its header bar instead and drives the active group via `group`. */
		hideTabs?: boolean;
		/** Externally-controlled active group (used with `hideTabs`). */
		group?: string | null;
	};
	const { node, showHeader = true, hideTabs = false, group: groupProp = null }: Props = $props();

	const g = graph();

	function setValue(group: string, name: string, value: unknown): void {
		if (!node) return;
		void g.updateParam(node.uid, group, name, value);
	}

	function setExpression(
		group: string,
		name: string,
		expression: string | null,
		opts: { enabled?: boolean; triggers_process?: boolean; autoeval?: boolean } = {}
	): void {
		if (!node) return;
		void g.setExpression(node.uid, group, name, expression, opts);
	}

	const groupNames = $derived(paramGroupNames(node));

	let internalTab = $state<string | null>(null);
	let docsOpen = $state(false);

	// --- inline name editing (slide-in inspector header) --------------------
	// The display `name` is the only mutable, display-only attribute (identity is
	// the uid). Names are flat and globally unique at every nesting depth — no
	// `inst::local` qualification — so the whole name is editable directly.
	// Keyed by uid so switching nodes auto-closes the editor while live state
	// updates (which re-create the node object) leave an open edit untouched.
	let editingUid = $state<string | null>(null);
	let nameDraft = $state('');
	const editingName = $derived(node != null && editingUid === node.uid);

	function startRename(): void {
		if (!node) return;
		nameDraft = node.name;
		editingUid = node.uid;
	}
	function commitRename(): void {
		// Escape/cancel nulls editingUid first, so the blur it triggers as the input
		// unmounts is a no-op here — only a live edit commits.
		const uid = editingUid;
		editingUid = null;
		if (!uid || !node || node.uid !== uid) return;
		const base = nameDraft.trim();
		if (!base) return; // empty → keep the current name
		void g.renameNode(uid, base);
	}
	function cancelRename(): void {
		editingUid = null;
	}
	function focusInput(el: HTMLInputElement): void {
		el.focus();
		el.select();
	}

	// When this component owns the tabs (inspector), keep the active group valid
	// as the node changes. When `hideTabs`, the parent controls it via `group`.
	$effect(() => {
		if (hideTabs) return;
		const valid = groupNames;
		if (valid.length === 0) {
			internalTab = null;
			return;
		}
		if (!internalTab || !valid.includes(internalTab)) {
			internalTab = valid[0];
		}
	});

	const activeTab = $derived(hideTabs ? groupProp : internalTab);

	const activeParams = $derived.by(() => {
		if (!node || !activeTab) return [] as [string, ParamDescriptor][];
		const group = node.params[activeTab] ?? {};
		return Object.entries(group) as [string, ParamDescriptor][];
	});
</script>

<section class="panel">
	{#if !node}
		<div class="empty">
			<div class="empty-title">No node selected</div>
			<div class="empty-sub">Click a node to edit its parameters.</div>
		</div>
	{:else if node.subpatch}
		<!-- A virtual sub-patch node: sharing controls + mirror list, not params. -->
		<SubPatchInspector {node} />
	{:else}
		{#if showHeader}
			<header class:has-doc={Boolean(node.doc)} class:expanded={docsOpen}>
				<span class="dot" style="background: {categoryColor(node.category)};"></span>
				<div class="titles">
					<div class="title">
						{#if editingName}
							<!-- svelte-ignore a11y_autofocus -->
							<input
								class="rename"
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
								class="base"
								title="Click to rename"
								onclick={startRename}
								data-testid="node-name">{node.name}</button
							>
						{/if}
					</div>
					<div class="sub">{formatName(node.type)}</div>
				</div>
				<span class="badge" class:badge-error={Boolean(node.error)} class:badge-ok={!node.error}>
					{node.error ? 'error' : 'running'}
				</span>
				{#if node.doc}
					<button
						class="docs-toggle"
						class:open={docsOpen}
						onclick={() => (docsOpen = !docsOpen)}
						title={docsOpen ? 'Hide docs' : 'Show docs'}
						aria-label={docsOpen ? 'Hide docs' : 'Show docs'}
						data-testid="docs-toggle"
					>
						▸
					</button>
				{/if}
			</header>

			{#if docsOpen && node.doc}
				<p class="docstring" data-testid="docstring">{node.doc}</p>
			{/if}
		{/if}

		{#if !hideTabs}
			<div class="tabs" role="tablist" data-testid="param-tabs">
				{#each groupNames as groupName (groupName)}
					<button
						class="tab"
						class:active={activeTab === groupName}
						role="tab"
						aria-selected={activeTab === groupName}
						onclick={() => (internalTab = groupName)}
					>
						{groupName}
					</button>
				{/each}
			</div>
		{/if}

		<div class="rows" role="tabpanel" data-testid="param-rows">
			{#if activeParams.length === 0}
				<div class="empty-tab">No parameters in this group.</div>
			{:else}
				{#each activeParams as [paramName, descriptor] (paramName)}
					<ParamField
						{paramName}
						{descriptor}
						onCommit={(v) => setValue(activeTab ?? '', paramName, v)}
						onSetExpression={(expr, opts) =>
							setExpression(activeTab ?? '', paramName, expr, opts)}
					/>
				{/each}
			{/if}
		</div>
	{/if}
</section>

<style>
	.panel {
		padding: 0;
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	.empty {
		padding: 36px 12px;
		text-align: center;
		color: var(--text-dim);
	}
	.empty-title {
		font-size: 13px;
		color: var(--text);
		margin-bottom: 4px;
	}
	.empty-sub {
		font-size: 11px;
	}
	header {
		display: flex;
		gap: 10px;
		align-items: center;
		padding: 10px 12px;
		border-bottom: 1px solid var(--border);
		background: var(--bg-elev-1);
	}
	.dot {
		width: 10px;
		height: 10px;
		border-radius: 50%;
		flex-shrink: 0;
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
	.base {
		font: inherit;
		color: var(--text);
		background: none;
		border: none;
		padding: 0;
		cursor: text;
		border-radius: var(--radius-sm);
	}
	.base:hover {
		text-decoration: underline;
		text-decoration-style: dotted;
		text-underline-offset: 2px;
	}
	.rename {
		width: 100%;
		font: inherit;
		font-size: 14px;
		font-weight: 600;
		padding: 1px 4px;
		color: var(--text);
		background: var(--bg-elev-2);
		border: 1px solid var(--accent);
		border-radius: var(--radius-sm);
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
		flex-shrink: 0;
	}
	.badge-ok {
		color: var(--success);
		background: color-mix(in srgb, var(--success) 14%, transparent);
		border: 1px solid color-mix(in srgb, var(--success) 30%, transparent);
	}
	.badge-error {
		color: var(--danger);
		background: color-mix(in srgb, var(--danger) 14%, transparent);
		border: 1px solid color-mix(in srgb, var(--danger) 35%, transparent);
	}
	.docs-toggle {
		font-family: var(--font-mono);
		font-size: 12px;
		color: var(--text-faint);
		background: none;
		border: none;
		padding: 2px 4px;
		cursor: pointer;
		transition:
			transform 100ms ease,
			color 100ms ease;
		flex-shrink: 0;
	}
	.docs-toggle:hover {
		color: var(--text);
	}
	.docs-toggle.open {
		transform: rotate(90deg);
		color: var(--accent);
	}
	.docstring {
		margin: 0;
		font-size: 11px;
		color: var(--text-dim);
		background: var(--bg-elev-2);
		padding: 8px 12px;
		white-space: pre-wrap;
		border-bottom: 1px solid var(--border);
	}
	.tabs {
		/* Tab strip — one tab per parameter group. Horizontally scrollable
		 * when many groups exist; the active tab's underline tracks via
		 * border-bottom on the button itself. */
		display: flex;
		gap: 0;
		overflow-x: auto;
		border-bottom: 1px solid var(--border);
		scrollbar-width: thin;
		background: var(--bg-elev-1);
	}
	.tab {
		flex: 0 0 auto;
		padding: 8px 14px;
		background: none;
		border: none;
		border-bottom: 2px solid transparent;
		color: var(--text-dim);
		font-family: var(--font-mono);
		font-size: 11px;
		letter-spacing: 0.03em;
		text-transform: lowercase;
		cursor: pointer;
		transition:
			color 80ms ease,
			border-color 80ms ease;
	}
	.tab:hover {
		color: var(--text);
	}
	.tab.active {
		color: var(--text);
		border-bottom-color: var(--accent);
	}
	.rows {
		display: flex;
		flex-direction: column;
		gap: 10px;
		padding: 12px;
	}
	.empty-tab {
		color: var(--text-faint);
		font-size: 11px;
		text-align: center;
		padding: 20px 0;
	}
</style>
