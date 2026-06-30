<script lang="ts">
	import ViewerFeed from './ViewerFeed.svelte';
	import ViewerControls from './ViewerControls.svelte';
	import { rawInlineView, setInlineKind, setInlineSetting } from './inlineView.svelte';
	import { recordViewChange } from './viewExecutors';
	import { resolveKind } from './kind';
	import { resolveSettings, type SettingsMap } from './settingsSchema';
	import type { ViewBinding } from './viewBinding';
	import { ui } from '$lib/stores/ui.svelte';
	import { graph } from '$lib/stores/graph.svelte';
	import { dtypeColor } from '$lib/editor/categoryColor';

	// `label` overrides the displayed slot name (the handle id stays `slot`): a sub-patch
	// synth node passes its renameable portal name so the collapsed output header matches
	// the In/Out pill. Absent for a real node → the slot id is shown.
	type Props = { node: string; slot: string; dtype: string; label?: string };
	const { node, slot, dtype, label }: Props = $props();

	const g = graph();
	const uiStore = ui();

	// The inline viewer's binding: backed by the node-scoped inline-view store,
	// persisting into node.viewers via pushNodeViewers. Built here (its single use
	// site) so viewBinding.ts stays rune-free and unit-testable.
	// Raw (pre-resolution) snapshot of this slot's view state, for undo capture.
	function snap(): { kind?: ReturnType<typeof resolveKind> | undefined; settings: SettingsMap } {
		const v = rawInlineView(node, slot);
		return { kind: v.kind, settings: { ...v.settings } };
	}
	const binding: ViewBinding = {
		get kind() {
			return resolveKind(dtype, rawInlineView(node, slot).kind);
		},
		get settings() {
			return resolveSettings(this.kind, rawInlineView(node, slot).settings);
		},
		setKind(k) {
			const before = snap();
			setInlineKind(node, slot, k);
			g.pushNodeViewers(node);
			recordViewChange({ kind: 'inline', node, slot }, before, snap(), `Viewer → ${k}`);
		},
		setSetting(key, value) {
			const before = snap();
			setInlineSetting(node, slot, key, value);
			g.pushNodeViewers(node);
			recordViewChange({ kind: 'inline', node, slot }, before, snap(), `Viewer ${key}`);
		}
	};

	function onSlotClick(e: MouseEvent): void {
		// Clicking the slot name opens the add-node menu seeded to wire a new node
		// onto this output — outputs fan out, so this never disconnects existing
		// cables. The matching connector pill (in GoofiNode's overlay) does the
		// same; dragging the pill starts a connection instead.
		e.stopPropagation();
		ui().requestSlotClick({ node, slot, dtype, side: 'source', clientX: e.clientX, clientY: e.clientY });
	}

	const expanded = $derived(uiStore.isSlotExpanded(node, slot));

	function toggleExpanded(e?: Event): void {
		// Stop the toggle from bubbling to SvelteFlow's node handlers, so
		// collapsing or expanding a viewer never selects (or grabs) the node.
		e?.stopPropagation();
		uiStore.toggleSlotExpanded(node, slot);
		// Collapse is the only inline view-state not owned by the binding; persist
		// it here at its mutation site (kind/settings persist via the binding).
		g.pushNodeViewers(node);
	}
	function stopSelect(e: Event): void {
		// SvelteFlow begins node selection/drag on pointerdown; keep that off the
		// slot header bar. Its children (kind dropdown, cog, slot name) still get
		// their own clicks — stopPropagation doesn't preventDefault.
		e.stopPropagation();
	}
</script>

<div
	class="slot-viewer"
	class:collapsed={!expanded}
	style="--dtype: {dtypeColor(dtype)};"
	data-node={node}
	data-slot={slot}
>
	<header
		onclick={toggleExpanded}
		onpointerdown={stopSelect}
		role="button"
		tabindex="0"
		aria-expanded={expanded}
	>
		<button class="tri" class:open={expanded} onclick={toggleExpanded} aria-label="toggle viewer">
			<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M4 2.5 L8.5 6 L4 9.5 Z" /></svg>
		</button>
		<span class="hspace"></span>
		{#if expanded}
			<ViewerControls {dtype} {binding} />
		{/if}
		<span
			class="slot-name"
			onclick={onSlotClick}
			role="button"
			tabindex="0"
			data-testid="slot-output"
			title={dtype.toLowerCase()}
		>
			{label ?? slot}
		</span>
	</header>

	{#if expanded}
		<div class="body"><ViewerFeed {node} {slot} {binding} /></div>
	{/if}
</div>

<style>
	.slot-viewer {
		display: flex;
		flex-direction: column;
		/* No background of its own — the header bar and the viewer body each draw
		   their own, so the slot-viewer box never overhangs the node's rounded
		   corners (which is what was clipping the bottom border). */
	}
	header {
		/* A collapsed slot is exactly one unit tall so slots stack on the node's
		   grid. A faint, dark tint of the slot's dtype colour sets the header bar
		   apart from the node body. The horizontal padding keeps the triangle and
		   the slot name clear of the input/output connectors in GoofiNode's
		   overlay. */
		height: var(--node-u);
		box-sizing: border-box;
		border-top: 1px solid var(--border);
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 0 12px 0 12px;
		background: color-mix(in srgb, var(--dtype, var(--text-dim)) 13%, var(--bg));
		font-size: 10px;
		cursor: pointer;
		user-select: none;
		transition: background 80ms ease;
	}
	header:hover {
		background: color-mix(in srgb, var(--dtype, var(--accent)) 20%, var(--bg));
	}

	/* A proper disclosure triangle: an SVG that rotates open/shut, vertically
	   centred, no button chrome. */
	.tri {
		display: inline-grid;
		place-items: center;
		width: 16px;
		height: 16px;
		padding: 0;
		background: none;
		border: 0;
		cursor: pointer;
		flex-shrink: 0;
	}
	.tri svg {
		width: 11px;
		height: 11px;
		fill: var(--text-dim);
		transform: rotate(0deg);
		transition:
			transform 120ms ease,
			fill 80ms ease;
	}
	.tri.open svg {
		transform: rotate(90deg);
	}
	.tri:hover svg {
		fill: var(--text);
	}

	.hspace {
		flex: 1 1 auto;
	}
	.slot-name {
		/* Sits right against the output connector so the label reads as its name. */
		font-family: var(--font-mono);
		color: var(--dtype, var(--text-dim));
		cursor: pointer;
		border-radius: 3px;
		padding: 0 2px;
		transition: background 80ms ease;
	}
	.slot-name:hover {
		background: color-mix(in srgb, var(--dtype, var(--accent)) 22%, transparent);
	}
	.body {
		height: var(--node-viewer);
		box-sizing: border-box;
		/* Keep the plot inside its slot; the node surface clips the outer corners. */
		overflow: hidden;
		display: flex;
		align-items: stretch;
		justify-content: stretch;
		padding: 4px 6px 7px;
		background: var(--bg);
	}
	.body > :global(*) {
		flex-grow: 1;
		min-width: 0;
		min-height: 0;
	}
</style>
