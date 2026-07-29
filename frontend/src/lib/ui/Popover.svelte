<!--
  Popover — one anchored, self-dismissing overlay (spec §2.4). It portals its content to <body>
  (escaping panel/SvelteFlow transform + clip contexts), positions it against the `anchor` element
  via the pure `clampToViewport` SSOT, and dismisses on Escape OR a pointerdown outside both the
  surface and the anchor (calling `onDismiss` — the parent owns the open state). Its product
  consumers are the error chip's list and the per-slot viewer settings menu; the two POINT-anchored
  menus (ContextMenu and the add-node menu) keep their own shells and share only the clamp — see
  D-M2, and each file's own note on why.

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
		catcher = false,
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
		/** Render a full-screen catcher under the surface so a dismissing click ONLY dismisses. */
		catcher?: boolean;
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
		const el = menuEl;
		const a = anchor;
		const place = (): void => {
			const ar = a.getBoundingClientRect();
			const m = el.getBoundingClientRect();
			pos = clampToViewport(
				ar,
				{ width: m.width, height: m.height },
				{ width: window.innerWidth, height: window.innerHeight },
				{ flip }
			);
			placed = true;
		};
		// Measured on every resize, not once per open: the surface's content is the consumer's and can
		// grow while it is open (the error list derives live from the control plane). With `flip` the
		// surface is pinned by its TOP, so growth extends DOWNWARD over the very trigger the flip
		// exists to keep clear. No feedback loop — repositioning a fixed element does not resize it.
		const ro = new ResizeObserver(place);
		ro.observe(el);
		place();
		return () => ro.disconnect();
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
	{#if catcher}
		<!-- Opt-in click catcher: a portalled full-screen layer one step under the surface. It carries
		     no handler, because it needs none — the window `pointerdown` above is what dismisses, and
		     it is armed exactly when this layer is mounted (both gated on `open`). What the layer does
		     is ABSORB the pointer event, so a dismissing click CONSUMES instead of also acting on
		     whatever it landed on. The window listener alone cannot do that: the handler under the
		     pointer still fires, and a target that stops propagation (the slot header, which must keep
		     SvelteFlow from starting a node drag) never lets the listener see the event at all. So the
		     job here is entirely the CSS below; it reads --popover-z from the portal root, exactly as
		     the surface does. (It once mirrored the add-node menu's overlay and carried an `onclick`
		     too — dead here: the dismissal unmounts this branch at the microtask checkpoint ending the
		     pointerdown task, so `click` is never dispatched to it. The add-node menu, which has no
		     window listener at all, still needs its own.) -->
		<div class="ui-popover-catcher" use:portal></div>
	{/if}
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
	.ui-popover-catcher {
		position: fixed;
		inset: 0;
		z-index: calc(var(--popover-z, var(--z-menu)) - 1);
	}
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
