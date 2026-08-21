<script lang="ts">
	import { onMount } from 'svelte';
	import { createLongPress } from 'panelty';
	import { clampToViewport, overlayViewport } from 'panelty';
	import { nearestTitle } from './titleTip';

	const LINGER_MS = 5000;

	let tip = $state<{ el: HTMLElement; text: string } | null>(null);
	let tipEl = $state<HTMLDivElement | null>(null);
	let pos = $state({ left: 0, top: 0 });
	// Hidden until the first measurement lands, so the bubble never flashes at (0,0).
	let placed = $state(false);

	let armed: { el: HTMLElement; text: string } | null = null;
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
		// Capture throughout: a control that stops propagation on pointerdown would otherwise never
		// be askable, and the click must be caught before its target to be swallowed at all.
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
		pointer-events: none;
		white-space: pre-wrap;
	}
</style>
