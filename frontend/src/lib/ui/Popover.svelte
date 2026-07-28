<!--
  Popover — one anchored, self-dismissing overlay (spec §2.4). The SSOT that replaces the ~10
  hand-rolled popovers (6 with their own re-implemented clamp): it portals its content to <body>
  (escaping panel/SvelteFlow transform + clip contexts), positions it against the `anchor` element
  via the pure `clampToViewport`, and dismisses on Escape OR a pointerdown outside both the surface
  and the anchor (calling `onDismiss` — the parent owns the open state).

  The anchor is excluded from "outside" so the trigger's own onclick toggles cleanly (an outside
  pointerdown that also hit the anchor would dismiss-then-reopen). Surface chrome is F tokens, each
  a `var(--popover-*, <token>)` per-instance hook (spec §1). `class` merged, `data-testid` (and any
  other attribute) forwarded via `...rest`.

  Semantics are the consumer's, not the primitive's: this is an unstyled-semantics positioned
  surface that imposes NO role of its own — the anchored, self-dismissing model, not the modal,
  focus-trapping `Dialog` (that primitive owns the name + focus context a `role="dialog"` demands).
  A consumer declares the fitting role/name (`role="menu"`, `aria-label`, …) via `...rest`, or lets
  the interactive children carry their own roles.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';
	import { portal } from '$lib/workspace/portal';
	import { clampToViewport } from './clampToViewport';

	let {
		anchor,
		open,
		onDismiss,
		flip = false,
		class: klass = '',
		children,
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		/** The element the popover hangs beneath; its rect drives the clamp. */
		anchor: HTMLElement | null;
		open: boolean;
		onDismiss: () => void;
		/** Let the surface flip ABOVE a bottom-anchored trigger rather than shift up and cover it. */
		flip?: boolean;
		children?: Snippet;
	} = $props();

	let menuEl = $state<HTMLDivElement | null>(null);
	let pos = $state<{ left: number; top: number }>({ left: 0, top: 0 });
	// Hidden until the first measurement lands, so the popover never flashes at (0,0) before the
	// clamp positions it (the element-anchored analogue of ContextMenu's known-spawn-point start).
	let placed = $state(false);

	$effect(() => {
		if (!open || !menuEl || !anchor) {
			placed = false;
			return;
		}
		const a = anchor.getBoundingClientRect();
		const m = menuEl.getBoundingClientRect();
		pos = clampToViewport(
			a,
			{ width: m.width, height: m.height },
			{ width: window.innerWidth, height: window.innerHeight },
			{ flip }
		);
		placed = true;
	});

	function onWindowPointerDown(e: PointerEvent): void {
		const t = e.target as Node | null;
		// Inside the surface, or on the anchor (its own onclick toggles) → not an outside dismiss.
		if (t && menuEl?.contains(t)) return;
		if (t && anchor?.contains(t)) return;
		onDismiss();
	}
	function onWindowKeydown(e: KeyboardEvent): void {
		if (e.key === 'Escape') onDismiss();
	}
</script>

<!-- svelte:window must be top-level; the handlers are nulled out while closed rather than the tag
     being conditionally rendered (mirrors ContextMenu). -->
<svelte:window
	onpointerdown={open ? onWindowPointerDown : undefined}
	onkeydown={open ? onWindowKeydown : undefined}
/>

{#if open}
	<div
		{...rest}
		bind:this={menuEl}
		class={`ui-popover ${klass}`.trim()}
		style="left:{pos.left}px; top:{pos.top}px; visibility:{placed ? 'visible' : 'hidden'}"
		use:portal
	>
		{@render children?.()}
	</div>
{/if}

<style>
	.ui-popover {
		position: fixed;
		z-index: var(--popover-z, var(--z-menu));
		min-width: var(--popover-min-width, 12rem);
		max-width: var(--popover-max-width, 90vw);
		padding: var(--popover-pad, var(--space-4));
		background: var(--popover-bg, var(--surface-2));
		border: var(--popover-border, 1px solid var(--border-strong));
		border-radius: var(--popover-radius, var(--radius-md));
		box-shadow: var(--popover-shadow, var(--shadow-2));
		color: var(--text);
		font-size: var(--fs-small);
	}
</style>
