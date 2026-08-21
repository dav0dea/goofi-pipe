<script lang="ts">
	import ViewerFeed from './ViewerFeed.svelte';
	import ViewerControls from './ViewerControls.svelte';
	import { slotView, isSlotExpanded } from './inlineView';
	import { recordViewChange } from './viewExecutors';
	import { resolveKind, type ViewerKind } from './kind';
	import { resolveSettings, type SettingsMap } from './settingsSchema';
	import type { ViewBinding } from './viewBinding';
	import { ui } from '$lib/stores/ui.svelte';
	import { graph } from '$lib/stores/graph.svelte';
	import { dtypeColor } from '$lib/editor/categoryColor';

	// `label` overrides the displayed slot name for a sub-patch portal; the handle id stays `slot`.
	type Props = { node: string; slot: string; dtype: string; label?: string };
	const { node, slot, dtype, label }: Props = $props();

	const g = graph();

	// Built here, its single use site, so viewBinding.ts stays rune-free.
	const rec = $derived(g.nodeById(node));
	// Raw (pre-resolution) snapshot of this slot's view state, for undo capture.
	function snap(): { kind?: ViewerKind; settings: SettingsMap } {
		const v = slotView(rec, slot);
		return { kind: v.kind, settings: { ...v.settings } };
	}
	const binding: ViewBinding = {
		get kind() {
			return resolveKind(dtype, slotView(rec, slot).kind);
		},
		get settings() {
			return resolveSettings(this.kind, slotView(rec, slot).settings);
		},
		setKind(k) {
			// The write round-trips through the document, so AFTER is the value asked for, not a re-read.
			const before = snap();
			const after = { ...before, kind: k };
			g.setSlotView(node, slot, after);
			recordViewChange({ kind: 'inline', node, slot }, before, after, `Viewer → ${k}`);
		},
		setSetting(key, value) {
			const before = snap();
			const after = { kind: before.kind, settings: { ...before.settings, [key]: value } };
			g.setSlotView(node, slot, after);
			recordViewChange({ kind: 'inline', node, slot }, before, after, `Viewer ${key}`);
		}
	};

	function onSlotClick(e: MouseEvent): void {
		// Opens the add-node menu seeded to wire onto this output; outputs fan out, so nothing disconnects.
		e.stopPropagation();
		ui().requestSlotClick({ node, slot, dtype, side: 'source', clientX: e.clientX, clientY: e.clientY });
	}

	const expanded = $derived(isSlotExpanded(rec, slot));

	function toggleExpanded(e?: Event): void {
		// Keep the toggle off SvelteFlow's node handlers, so a collapse never selects the node.
		e?.stopPropagation();
		g.setSlotView(node, slot, { collapsed: expanded });
	}
	function stopSelect(e: PointerEvent): void {
		// Keeps the press off the window-level bubble listeners (`Popover`'s outside-press dismiss).
		// Released for TOUCH: a viewer is most of a node's surface, so a finger there is reaching for the node.
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
	<!-- The header bar is the pointer target for collapse/expand; keyboard uses the ► button. -->
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
		<!-- Pointer convenience: opens the add-node menu at the cursor. -->
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
		/* No background of its own, so the box never overhangs the node's rounded corners. */
	}
	header {
		/* Exactly one unit tall, so slots stack on the node's grid. */
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
	/* GoofiNode's `.header` paints its own `border-bottom`, so a first slot would double the seam. */
	.slot-viewer:first-child header {
		border-top: none;
	}
	/* Hover chrome gated on real hover: `:hover` still MATCHES on a phone, for a state a finger
	   is never in. A hover-CAPABILITY query, deliberately not a `pointer` one. */
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

	/* `color: inherit` because the UA colours a bare <button> `buttontext`, invisible under a fill-ed SVG. */
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

	/* This header is one `--node-u` and `.surface` clips it, so nothing inside may take the 44px
	   coarse floor: the header BAR takes the tap instead. */
	@media (hover: none) and (pointer: coarse) {
		header {
			--select-min-h: 0;
			--select-fs: var(--fs-small);
			/* The cog keeps its 16px paint for the same reason; a `::after` carries its coarse target outward. */
			--vs-cog-box: 16px;
		}
		.tri {
			min-height: 0;
		}
	}
</style>
