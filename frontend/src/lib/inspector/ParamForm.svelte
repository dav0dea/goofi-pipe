<script module lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';
	import type { ParamDescriptor } from '$lib/api/types';
	import type { StatusTone } from '$lib/ui';
	import type { BadgeTone } from '$lib/ui/Badge.svelte';

	/** The health scale onto the Badge's — two tone vocabularies, mapped in one place. */
	const BADGE_TONE: Record<StatusTone, BadgeTone> = {
		ok: 'success',
		warn: 'warning',
		error: 'danger'
	};

	/** The param count a node needs before it is worth offering a filter. */
	export const SEARCH_FROM = 8;

	/** …and before browsing is hopeless enough to want a touched-only filter as well. A hand-written
	    node is read; a plugin's hundreds are hunted, and only the second wants the switch. */
	export const TOUCHED_FROM = 40;

</script>

<!--
  ParamForm — the node-driven inspector: an identity header, the group tab strip it owns, and one
  `<ParamField>` per param in the active group.
-->
<script lang="ts">
	import type { SourcePatch } from '$lib/api/types';
	import type { HTMLAttributes } from 'svelte/elements';
	import { graph } from '$lib/stores/graph.svelte';
	import { isValidName } from '$lib/crdt/graphDoc';
	import { formatName } from '$lib/editor/categoryColor';
	import { bareName } from '$lib/editor/typeId';
	import { nodeHealth } from '$lib/editor/nodeHealth';
	import ParamField from './ParamField.svelte';
	import SubPatchInspector from '$lib/editor/SubPatchInspector.svelte';
	import { matchParams, type ParamHit } from './paramSearch';
	import { onlyTouched, touchedCount, touchedRows } from './paramTouched';
	import { Bar, Tabs, Badge, Disclosure, EmptyState, Icon, IconButton, MODE_ATTRS, Toggle } from '$lib/ui';

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
	function pulse(group: string, name: string): void {
		if (!node) return;
		void g.pulse(node.uid, group, name).catch((e) => console.warn('pulse failed', e));
	}
	function setSource(group: string, name: string, source: SourcePatch): void {
		if (!node) return;
		void g.setSource(node.uid, group, name, source).catch((e) => console.warn('set source failed', e));
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
		// The manager refuses a name an expression could not read as an attribute; saying so here
		// is what keeps the draft rather than throwing the user's typing away on a blur.
		if (!isValidName(base)) return;
		void g.renameNode(uid, base).catch((e) => console.warn('rename failed', e));
	}
	function cancelRename(): void {
		editingUid = null;
	}
	function focusInput(el: HTMLInputElement): void {
		el.focus();
		el.select();
	}

	const groupNames = $derived(node ? Object.keys(node.params) : []);
	const health = $derived(nodeHealth(node));

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

	// One search box over every family, because a plugin's parameters are spread across as many tabs
	// as it has units and the one you want is rarely in the tab you are on.
	let query = $state('');
	const searching = $derived(query.trim().length > 0);
	// A short node is faster to read than to filter; a plugin with no units is one long tab.
	const paramCount = $derived(
		Object.values(node?.params ?? {}).reduce((n, named) => n + Object.keys(named ?? {}).length, 0)
	);
	const searchable = $derived(paramCount > SEARCH_FROM);
	const filterable = $derived(paramCount > TOUCHED_FROM);

	// A plugin declares every param it has, so browsing one is a scroll; touched-only is the way
	// through. It starts OFF and returns there whenever the selection moves: the control only shows
	// on a node with params enough to need it, so a filter carried onto a small node would hide its
	// params behind a switch that is not on screen to turn off.
	let touchedOnly = $state(false);
	let filtered = $state<string | null>(null);
	$effect(() => {
		const uid = node?.uid ?? null;
		if (uid !== filtered) {
			filtered = uid;
			touchedOnly = false;
		}
	});
	const touched = $derived(touchedCount(node?.params));

	/** True while the list spans every group rather than the fronted tab. */
	const across = $derived(searching || touchedOnly);

	// All three modes reduce to the same row list, so a field is rendered from one place whichever
	// is on. Touched-only spans EVERY group: a knob was turned in the plugin's own window, and
	// which tab goofi filed it under is the one thing the reader does not know.
	const rows = $derived.by<ParamHit[]>(() => {
		const n = node;
		if (!n) return [];
		// A search inside the filter searches what the filter admits: the toggle is the standing
		// question, and a query narrows that rather than reopening everything behind it.
		if (searching) {
			const hits = matchParams(n.params, query);
			return touchedOnly ? onlyTouched(hits) : hits;
		}
		const named = (g: string) => (n.params[g] ?? {}) as Record<string, ParamDescriptor>;
		const of = (g: string) =>
			Object.entries(named(g)).map(([name, descriptor]) => ({ group: g, name, descriptor }));
		if (touchedOnly) return touchedRows(n.params, groupNames);
		return activeGroup ? of(activeGroup) : [];
	});
</script>

<section {...rest} class={`param-form ${klass}`.trim()}>
	{#if !node}
		<EmptyState data-testid="param-empty">
			{#snippet title()}No node selected{/snippet}
			{#snippet hint()}Select a node to edit its parameters.{/snippet}
		</EmptyState>
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
									class:bad={nameDraft.trim() !== '' && !isValidName(nameDraft.trim())}
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
						<div class="pf-type">{formatName(bareName(node.type))}</div>
					</div>
				{/snippet}
				{#snippet end()}
					<Badge
						tone={BADGE_TONE[health.tone]}
						class="pf-state"
						title={health.hint}
						data-testid="node-state"
					>
						{health.status}{#if health.runtime}<span class="pf-runtime" data-testid="node-runtime"
								>{health.runtime}</span
							>{/if}
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

		{#if node.subpatch}
			<SubPatchInspector {node} />
		{:else}
			{#if searchable || searching}
				<!-- Native, not `TextInput`: this filters per keystroke and owns Escape. -->
				<input
					class="pf-search"
					{...MODE_ATTRS.search}
					bind:value={query}
					onkeydown={(e) => {
						if (e.key === 'Escape') query = '';
					}}
					placeholder={touchedOnly ? 'Search touched…' : 'Search parameters…'}
					autocomplete="off"
					aria-label="Search parameters"
					data-testid="param-search"
				/>
			{/if}

			{#if filterable}
				<label class="pf-touched" data-testid="param-touched-only">
					<Toggle value={touchedOnly} onChange={(v) => (touchedOnly = v)} />
					<span>Touched only</span>
					<span class="pf-touched-count">{touched}</span>
				</label>
			{/if}

			{#if tabItems.length > 0 && !across}
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
				role={tabItems.length > 0 && !across ? 'tabpanel' : undefined}
				aria-label={across ? undefined : (activeGroup ?? undefined)}
				data-testid="param-rows"
			>
				{#if rows.length === 0}
					<div class="pf-empty-group" data-testid={searching ? 'param-no-matches' : 'param-empty-group'}>
						{#if searching}{touchedOnly ? 'No touched parameters match.' : 'No parameters match.'}{:else if touchedOnly}Nothing touched yet — move a control here or in the plugin's own window.{:else}No parameters in this group.{/if}
					</div>
				{:else}
					{#each rows as { group, name: paramName, descriptor } (node.uid + '/' + group + '/' + paramName)}
						<div class="pf-row">
							{#if across}
								<span class="pf-row-group" data-testid={`param-hit-group-${paramName}`}>{group}</span>
							{/if}
							<ParamField
								{paramName}
								selfName={node?.name}
								{descriptor}
								data-testid={`param-field-${paramName}`}
								refreshing={node != null && g.isRefreshing(node.uid, group, paramName)}
								onCommit={(v) => setValue(group, paramName, v)}
								onSetSource={(source) => setSource(group, paramName, source)}
								onRefresh={() => refreshOptions(group, paramName)}
								onPulse={() => pulse(group, paramName)}
							/>
						</div>
					{/each}
				{/if}
			</div>
		{/if}
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
	.pf-rename.bad {
		color: var(--danger);
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
	/* The runtime rides INSIDE the state pill rather than beside it: one pill is what the row has
	   space for, and a second would be the first thing a narrow pane drops. */
	.param-form :global(.pf-state .pf-runtime) {
		margin-inline-start: var(--space-2);
		padding-inline-start: var(--space-2);
		border-inline-start: 1px solid currentColor;
		opacity: 0.7;
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
	.pf-touched {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		margin: var(--space-3) var(--space-6) 0;
		color: var(--text-2);
		cursor: pointer;
	}
	.pf-touched-count {
		margin-left: auto;
		color: var(--text-muted);
		font-variant-numeric: tabular-nums;
	}

	.pf-search {
		font: inherit;
		color: var(--text-1);
		background: var(--surface-2);
		border: 1px solid var(--border-1);
		border-radius: var(--radius-2);
		padding: var(--space-2) var(--space-3);
		margin: var(--space-3) var(--space-6) 0;
		min-width: 0;
	}
	.pf-search::placeholder {
		color: var(--text-muted);
	}
	.pf-row {
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
		min-width: 0;
	}
	.pf-row-group {
		color: var(--text-muted);
		font-size: var(--fs-small);
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
