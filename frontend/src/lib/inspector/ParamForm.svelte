<script module lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';

	/** Parameter group names in display order: node-specific groups alphabetical,
	 * 'common' last. Exported as module API (spec §6) and used internally to build the
	 * tab strip — ParamForm now owns the tabs, so both mounts share this one ordering. */
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
  ParamForm — the node-driven form assembler (spec §4, D-N6; the rebuilt ParamPanel). A handful of P
  primitives assemble the inspector: an identity header (P `Bar` + inline rename + `Badge` state + a
  docs `Disclosure`), the connected P `Tabs` bar it now OWNS (the old hideTabs/external-group split is
  dropped — both mounts get the same strip), and one `<ParamField>` per param in the active group.

  `show_when` (spec §5) filters fields by another param's live value: a field whose predicate evaluates
  false is OMITTED, and a group whose EVERY field is thus hidden renders NO tab (never a live tab over
  an all-hidden group). `liveValues` spans ALL groups because a dependency may cross them; it is rebuilt
  reactively so committing a controller shows/hides its dependents live. A genuinely empty group (no
  fields at all) keeps its tab + the empty-group message.

  Identity-blind rows: each `<ParamField>` is wired to the store via closures over `(activeGroup, name)`.
  The store contract is preserved verbatim (`updateParam`/`setExpression`/`refreshParam`/`isRefreshing`/
  `renameNode`), as is the rename null-first commit/cancel dance and the `node.subpatch` delegation.
  `class`/`data-testid` forward to the root via `...rest`.
-->
<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import type { ParamDescriptor } from '$lib/api/types';
	import { graph } from '$lib/stores/graph.svelte';
	import { formatName } from '$lib/editor/categoryColor';
	import { evalShowWhen, type ShowWhenPredicate } from './showWhen';
	import ParamField from './ParamField.svelte';
	import SubPatchInspector from '$lib/editor/SubPatchInspector.svelte';
	import { Bar, Tabs, Badge, Disclosure, EmptyState, MODE_ATTRS } from '$lib/ui';

	let {
		node,
		showHeader = true,
		showWhen = {},
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLElement> & {
		node: NodeInstanceInfo | null;
		/** Show the identity header (rename + state + docs). True in the editor's slide-in inspector;
		 * false in the dedicated Parameters panel, which already names the node in its linkbar. */
		showHeader?: boolean;
		/** Declarative field-dependency rules, keyed by param NAME (spec §5). Absent → the field
		 * always shows.
		 *
		 * Nothing supplies these in production, and that is the shipped state, not an omission: no
		 * backend descriptor carries a rule (`grep -rn show_when crates/ nodes/` is empty) and the
		 * node library is Oscillator + Buffer, neither of which has a dependent parameter. Both
		 * product mounts pass nothing; `/dev/inspector` is the only supplier, and
		 * `inspector-gallery.spec.ts` drives it to prove the mechanism live. D-N5: ship the algebra
		 * and its proof ahead of a consumer, the way P shipped primitives. The node-type registry an
		 * earlier draft of this line named was specced and never built. */
		showWhen?: Record<string, ShowWhenPredicate>;
	} = $props();

	const g = graph();

	// The store seam — closures over the node uid; each RPC is fire-and-forget with a logged failure
	// (matching the refresh idiom) so a rejected call never surfaces as an unhandled rejection.
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

	// --- inline name editing (header) --------------------------------------------------------------
	// The display `name` is the only mutable, display-only attribute (identity is the uid). Keyed by
	// uid so switching nodes auto-closes the editor while live state updates (which re-create the node
	// object) leave an open edit untouched.
	let editingUid = $state<string | null>(null);
	let nameDraft = $state('');
	const editingName = $derived(node != null && editingUid === node.uid);

	function startRename(): void {
		if (!node) return;
		nameDraft = node.name;
		editingUid = node.uid;
	}
	function commitRename(): void {
		// Escape/cancel nulls editingUid first, so the blur it triggers as the input unmounts is a
		// no-op here — only a live edit commits.
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

	// --- show_when + group tabs --------------------------------------------------------------------
	// Live param values across ALL groups (dependencies may cross groups); rebuilt reactively so a
	// committed controller re-evaluates its dependents.
	const liveValues = $derived.by<Record<string, unknown>>(() => {
		const out: Record<string, unknown> = {};
		const n = node;
		if (!n) return out;
		for (const group of Object.values(n.params)) {
			for (const [name, d] of Object.entries(group)) {
				out[name] = (d as ParamDescriptor).value;
			}
		}
		return out;
	});

	function isVisible(name: string): boolean {
		const pred = showWhen[name];
		return pred ? evalShowWhen(pred, liveValues) : true;
	}

	const groupNames = $derived(paramGroupNames(node));

	// Groups that render a tab: one with ≥1 visible field, OR a genuinely empty group (no fields at
	// all — keeps its tab + empty-group message). A group whose every field is hidden shows no tab.
	const visibleGroupNames = $derived.by<string[]>(() => {
		const n = node;
		if (!n) return [];
		return groupNames.filter((gName) => {
			const entries = Object.entries(n.params[gName] ?? {});
			if (entries.length === 0) return true;
			return entries.some(([name]) => isVisible(name));
		});
	});

	const tabItems = $derived(visibleGroupNames.map((name) => ({ id: name, label: name })));

	// This component owns the tabs. The active group is DERIVED — the group in front when it is still
	// visible, else the first visible one, else none — and an effect then ADOPTS that fallback.
	//
	// Both halves earn their place. It used to be an `$effect` assigning `activeGroup` from `null`,
	// which is why the strip flashed on every open: an effect runs after the first paint, so the
	// initial render had no `.active` tab, and the class then flipped INTO `.ui-tab`'s `background`
	// transition — animating --surface-2 → --surface-1. Deriving it puts the right tab in the first
	// paint, so there is no transition to animate. But a pure derivation is not enough: groups sort
	// alphabetically, so when a hidden group reappears (`filter` before `signal`) `valid[0]` would
	// steal the front from a still-valid group. Adopting the fallback keeps it sticky — the front
	// group only moves when it actually vanishes.
	let frontGroup = $state<string | null>(null);
	const activeGroup = $derived.by<string | null>(() => {
		const valid = visibleGroupNames;
		if (valid.length === 0) return null;
		return frontGroup && valid.includes(frontGroup) ? frontGroup : valid[0];
	});
	$effect(() => {
		if (activeGroup !== frontGroup) frontGroup = activeGroup;
	});

	const activeParams = $derived.by<[string, ParamDescriptor][]>(() => {
		const n = node;
		if (!n || !activeGroup) return [];
		const group = n.params[activeGroup] ?? {};
		return (Object.entries(group) as [string, ParamDescriptor][]).filter(([name]) => isVisible(name));
	});
</script>

<section {...rest} class={`param-form ${klass}`.trim()}>
	{#if !node}
		<EmptyState data-testid="param-empty">
			{#snippet title()}No node selected{/snippet}
			{#snippet hint()}Select a node to edit its parameters.{/snippet}
		</EmptyState>
	{:else if node.subpatch}
		<!-- A virtual sub-patch node: sharing controls + mirror list, not params. -->
		<SubPatchInspector {node} />
	{:else}
		{#if showHeader}
			<Bar>
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
					<Badge tone={node.error ? 'danger' : 'success'} data-testid="node-state">
						{node.error ? 'error' : 'running'}
					</Badge>
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

		<!-- The panel is a tabpanel only when a tablist exists (a zero-param node renders no tabs — an
		     orphaned `tabpanel` role would have no owning tablist); named by its active group. -->
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
	/* The name reads as text until clicked; the dotted underline on hover hints it is editable. */
	.pf-name {
		font: inherit;
		color: var(--text);
		background: none;
		border: none;
		padding: 0;
		cursor: text;
		border-radius: var(--radius-sm);
	}
	.pf-name:hover {
		text-decoration: underline;
		text-decoration-style: dotted;
		text-underline-offset: 2px;
	}
	.pf-rename {
		width: 100%;
		font: inherit;
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
	/* The active tab drops to --surface-1 (the Tabs `--tabs-body` default); the rows paint the SAME
	   surface so the connected tab merges into the body flush beneath it — one piece, no seam line. */
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
	/* Touch: the name's only editable cue is a hover underline, invisible on a device with no hover
	   — give it a resting dotted underline so it reads as editable without one. Gated on the app's
	   single coarse idiom (D-R7): the cue exists BECAUSE hover is unavailable, so a hover-capable
	   touchscreen neither needs it nor should carry it. */
	@media (hover: none) and (pointer: coarse) {
		.pf-name {
			text-decoration: underline;
			text-decoration-style: dotted;
			text-underline-offset: 2px;
		}
	}
	/* Touch: lift the rename input to 16px so focusing it does not force-zoom iOS (the desktop
	   --fs-strong size below 16px is kept). Mirrors app.css's coarse input/textarea floor. */
	@media (hover: none) and (pointer: coarse) {
		.pf-rename {
			font-size: 16px;
		}
	}
</style>
