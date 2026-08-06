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
	function stopSelect(e: PointerEvent): void {
		// Keeps a press on the header bar off the WINDOW-level bubble listeners the canvas sits
		// under (`Popover`'s outside-press dismiss is the one that matters — see
		// viewer-settings.spec). It does NOT keep SvelteFlow from starting a node drag, whatever
		// this file used to claim: `@xyflow/svelte` drags by d3-drag, which binds `mousedown` and
		// `touchstart`, neither of which a `pointerdown` handler can reach.
		//
		// Released for TOUCH — the live `pointerType`, D-R2's gate, not a media query. A viewer is
		// most of a node's surface, so a finger landing on one is reaching for the NODE, and the
		// header has to hand it the same press its body would. A mouse keeps the desktop path
		// byte-for-byte; children (kind dropdown, cog, slot name) still get their own clicks either
		// way, since stopPropagation doesn't preventDefault.
		if (e.pointerType === 'touch') return;
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
	<!-- The whole header bar is a pointer click-target for collapse/expand;
	     keyboard users toggle it via the ► button it contains. -->
	<!-- svelte-ignore a11y_click_events_have_key_events -->
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
		<!-- Pointer convenience: opens the add-node menu at the cursor (like the
		     connector pill). Keyboard "add node" is the topbar + menu. -->
		<!-- svelte-ignore a11y_click_events_have_key_events -->
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
		transition: background var(--dur-fast) var(--ease);
	}
	/* The first slot sits directly under GoofiNode's `.header`, which paints its own
	   `border-bottom` — so this row's `border-top` abutted it with no margin and the seam rendered
	   at 2 CSS px, against the node card's own 1px outline. Asked from THIS side rather than the
	   parent's, so the rule stays inside the component that owns the line: `PlacementPreview`,
	   documented as mirroring GoofiNode exactly, already ships the same de-duplication as
	   `.viewers .slot-row:first-child`. GoofiNode's `border-bottom` deliberately stays — `inputUnits`
	   floors every body at one unit, so a node with NO outputs has no slot row to carry the line.
	   Geometry-neutral: the header is `box-sizing: border-box` at a fixed `--node-u`. */
	.slot-viewer:first-child header {
		border-top: none;
	}
	/* The viewer's hover chrome, gated on the device actually HAVING a hover.
	   `:hover` still MATCHES on a phone — the browser resolves it for whatever synthetic pointer
	   passes over — so without this the header tinted, the triangle brightened and the slot name
	   washed for a state a finger can never be in, and each one had to be un-painted again on the
	   press that followed. This is a hover-CAPABILITY query, not a pointer one: D-R7's single coarse
	   idiom answers "is this a touch target" and is untouched (`theme/styleDrift.test.ts` scans
	   every prelude that mentions `pointer`, and this one deliberately does not). */
	@media (hover: hover) {
		header:hover {
			background: color-mix(in srgb, var(--dtype, var(--accent)) 20%, var(--bg));
		}
		.tri:hover svg {
			fill: var(--text);
		}
		.slot-name:hover {
			background: color-mix(in srgb, var(--dtype, var(--accent)) 22%, transparent);
		}
	}

	/* A proper disclosure triangle: an SVG that rotates open/shut, vertically
	   centred, no button chrome. `color: inherit` because a <button> is coloured
	   `buttontext` (white) by the UA now that app.css's base skin is gone — invisible
	   while the only content is a `fill:`-ed SVG, and a trap the moment it isn't. */
	.tri {
		display: inline-grid;
		place-items: center;
		width: 16px;
		height: 16px;
		padding: 0;
		background: none;
		border: 0;
		color: inherit;
		cursor: pointer;
		flex-shrink: 0;
	}
	.tri svg {
		width: 11px;
		height: 11px;
		fill: var(--text-dim);
		transform: rotate(0deg);
		transition:
			transform var(--dur-slow) var(--ease),
			fill var(--dur-fast) var(--ease);
	}
	.tri.open svg {
		transform: rotate(90deg);
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
		transition: background var(--dur-fast) var(--ease);
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

	/* Touch, and the one place in this file where the frozen canvas has to say so out loud. This
	   header is exactly one `--node-u` (24px) and `.surface` clips it, so nothing inside it can take
	   the 44px control floor — a floored control is not a bigger target here, it is a control
	   hanging out of the node with its bottom half cut off.
	   `.tri` states `height`, not `min-height`, so app.css's coarse floor DID apply and did exactly
	   that. Released, and the viewer-kind select re-pinned through ViewerControls' documented hook.
	   The stated exception: both keep their fine-pointer boxes on touch. What actually takes the tap
	   is the header BAR — full node width, one unit tall, and already the collapse target — and the
	   canvas is a pinch-zoomable surface, which no other strip in the app is. */
	@media (hover: none) and (pointer: coarse) {
		header {
			--viewer-kind-min-h: 0;
			--viewer-kind-fs: var(--fs-small);
			/* …and the cog keeps its frozen 16px paint here for the same reason (its coarse target is
			   carried outward by a `::after` bounded to this header instead). Every other host — a
			   docked ViewerPanel header is the one there is — floors the box. */
			--vs-cog-box: 16px;
		}
		.tri {
			min-height: 0;
		}
	}
</style>
