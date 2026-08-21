<!-- Popover — an anchored, self-dismissing overlay portalled to <body> and clamped to the
     viewport. It imposes no role of its own; the consumer declares one through `...rest`. -->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';
	import { portal, clampToViewport, overlayViewport } from 'panelty';

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
	// Hidden until the first measurement lands, so nothing flashes at (0,0).
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
			pos = clampToViewport(ar, { width: m.width, height: m.height }, overlayViewport(), { flip });
			placed = true;
		};
		// Re-measured on every resize, not once per open: the consumer's content can grow while open.
		const ro = new ResizeObserver(place);
		ro.observe(el);
		// …and on every visual-viewport change: a soft keyboard shrinks the space without resizing us.
		const vv = window.visualViewport;
		vv?.addEventListener('resize', place);
		place();
		return () => {
			ro.disconnect();
			vv?.removeEventListener('resize', place);
		};
	});

	function onWindowPointerDown(e: PointerEvent): void {
		const t = e.target as Node | null;
		// The anchor is not "outside": its own onclick toggles, so a dismiss here would reopen.
		if (t && menuEl?.contains(t)) return;
		if (t && anchor?.contains(t)) return;
		onDismiss();
	}
	function onWindowKeydown(e: KeyboardEvent): void {
		if (e.key !== 'Escape') return;
		// Consumed, and capture-phase for it: the open surface is topmost, so no window-bubble
		// listener beneath it may also act on this Escape.
		e.stopPropagation();
		onDismiss();
	}
</script>

<svelte:window
	onpointerdown={open ? onWindowPointerDown : undefined}
	onkeydowncapture={open ? onWindowKeydown : undefined}
/>

{#if open}
	{#if catcher}
		<!-- Handler-free by design: the layer only ABSORBS the pointer event, so a dismissing click
		     does not also act on what it landed on. The window listener above dismisses. -->
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
