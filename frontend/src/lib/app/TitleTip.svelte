<!--
  TitleTip — the app's one coarse-pointer door onto `title=` (R spec §3.1d, item A9).

  A `title` is a hover tooltip, and CLAUDE.md forbids an interaction or a piece of information
  existing solely behind hover. Roughly forty of them are in the tree and a dozen carry text that
  is rendered NOWHERE else: `AddNodeMenu`'s "missing dependency: …" (without which a greyed palette
  node cannot explain itself, breaking the contract CLAUDE.md states for it), a connector pill's
  dtype, `BoundaryNode`'s "Double-click to rename". So this is deliberately ONE layer rather than a
  resting cue per site — every existing tooltip gains the door at once, and every future one is born
  with it.

  Long-press is the idiom because it is the only gesture a touch UI has spare: a tap is the
  control's own action, and a press that has already been held past the recognizer's window is not
  going to become one. The press that opened a bubble therefore does NOT also fire the control —
  the click ending it is swallowed — which is what makes asking "what is this?" safe on a button
  that deletes something.

  Fine pointers are untouched: a mouse press is the start of a drag, never a request for help, and
  a mouse already HAS the native tooltip on hover. The gate is the live `pointerType`, not a media
  query, so a touchscreen laptop's finger gets the door while its mouse keeps the desktop path —
  one system, two representations (D-R2).
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { createLongPress } from '$lib/editor/longPress';
	import { clampToViewport, overlayViewport } from '$lib/gesture';
	import { nearestTitle } from './titleTip';

	/** How long a bubble nobody dismisses stays up. Long enough to read a dependency list. */
	const LINGER_MS = 5000;

	let tip = $state<{ el: HTMLElement; text: string } | null>(null);
	let tipEl = $state<HTMLDivElement | null>(null);
	let pos = $state({ left: 0, top: 0 });
	// Hidden until the first measurement lands, so the bubble never flashes at (0,0) before the
	// clamp positions it — Popover's idiom, for the same reason.
	let placed = $state(false);

	/** Resolved at pointerdown, promoted to `tip` only if the press survives the hold. */
	let armed: { el: HTMLElement; text: string } | null = null;
	/** Set when a press fires: the `click` ending that same press must not also act. */
	let swallowClick = false;
	let lingerTimer: ReturnType<typeof setTimeout> | null = null;

	const press = createLongPress(() => {
		if (!armed) return;
		tip = armed;
		swallowClick = true;
		lingerTimer = setTimeout(hide, LINGER_MS);
	});

	function hide(): void {
		tip = null;
		placed = false;
		if (lingerTimer) clearTimeout(lingerTimer);
		lingerTimer = null;
	}

	function onPointerDown(e: PointerEvent): void {
		hide();
		press.cancel();
		armed = null;
		swallowClick = false;
		if (e.pointerType === 'mouse') return;
		const hit = nearestTitle(e.target as HTMLElement | null);
		if (!hit) return;
		armed = { el: hit.el as HTMLElement, text: hit.text };
		press.start(e);
	}

	function onClick(e: MouseEvent): void {
		if (!swallowClick) return;
		swallowClick = false;
		e.preventDefault();
		e.stopPropagation();
	}

	onMount(() => {
		// Capture phase throughout: a control that stops propagation on its own pointerdown (the slot
		// header does, to keep SvelteFlow from starting a node drag) would otherwise never be
		// askable — and the swallowed click has to be caught before it reaches its target to be
		// swallowed at all. `scroll` only fires in capture at the window for a nested scroller.
		const opts = { capture: true } as const;
		window.addEventListener('pointerdown', onPointerDown, opts);
		window.addEventListener('pointermove', press.move, opts);
		window.addEventListener('pointerup', press.cancel, opts);
		window.addEventListener('pointercancel', press.cancel, opts);
		window.addEventListener('click', onClick, opts);
		window.addEventListener('scroll', hide, opts);
		return () => {
			window.removeEventListener('pointerdown', onPointerDown, opts);
			window.removeEventListener('pointermove', press.move, opts);
			window.removeEventListener('pointerup', press.cancel, opts);
			window.removeEventListener('pointercancel', press.cancel, opts);
			window.removeEventListener('click', onClick, opts);
			window.removeEventListener('scroll', hide, opts);
			press.cancel(); // a press in flight must not fire into an unmounted layer
			if (lingerTimer) clearTimeout(lingerTimer);
		};
	});

	// Placement: the pressed control is the anchor, and `flip` keeps the bubble off it — a tooltip
	// that covers the thing it describes (and the finger still on it) answers nothing.
	$effect(() => {
		const t = tip;
		const el = tipEl;
		if (!t || !el) return;
		const place = (): void => {
			const m = el.getBoundingClientRect();
			pos = clampToViewport(
				t.el.getBoundingClientRect(),
				{ width: m.width, height: m.height },
				overlayViewport(),
				{ flip: true }
			);
			placed = true;
		};
		place();
		const vv = window.visualViewport;
		vv?.addEventListener('resize', place);
		return () => vv?.removeEventListener('resize', place);
	});
</script>

{#if tip}
	<div
		class="title-tip"
		role="tooltip"
		bind:this={tipEl}
		data-testid="title-tip"
		style="left: {pos.left}px; top: {pos.top}px; visibility: {placed ? 'visible' : 'hidden'}"
	>
		{tip.text}
	</div>
{/if}

<style>
	.title-tip {
		position: fixed;
		z-index: var(--z-toast);
		max-width: min(24rem, 90vw);
		padding: var(--space-3) var(--space-5);
		background: var(--surface-2);
		border: 1px solid var(--border-strong);
		border-radius: var(--radius-sm);
		box-shadow: var(--shadow-2);
		color: var(--text);
		font-size: var(--fs-small);
		/* Inert by construction: the bubble is an answer, not a control, and the window pointerdown
		   above dismisses it wherever the next press lands — including on the bubble itself. */
		pointer-events: none;
		white-space: pre-wrap;
	}
</style>
