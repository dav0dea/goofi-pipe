<!--
  Dialog — a centered modal overlay (spec §2.4). A SEPARATE primitive from `Popover` (audit Q12:
  a centered/focus-trapped modal does not fit the anchored self-dismissing model). Built on the
  native `<dialog>` element via `showModal()`, which gives the three modal contracts for free and
  correctly: it promotes the dialog to the top layer (escaping every ancestor transform/clip with
  no portal), renders the `::backdrop`, TRAPS Tab focus within the dialog and moves focus into it
  on open, and routes Escape through the `cancel` event.

  The parent owns `open` (the SSOT): an `$effect` syncs the element's modal state to it, and every
  user dismissal — Escape (`cancel`) or a backdrop click — routes to `onClose` so the parent flips
  `open` (never the element behind its back, which would desync). A backdrop click is one that
  targets the dialog element itself (a click on content targets a child) AND lands outside its border
  box — the coordinate half matters because the dialog's own scrollbar is targeted exactly like the
  backdrop is. Surface chrome is F tokens via `var(--dialog-*, <token>)` hooks. `class` merged,
  `data-testid` (and any other attribute) forwarded via `...rest`.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';

	let {
		open,
		onClose,
		class: klass = '',
		children,
		...rest
	}: HTMLAttributes<HTMLDialogElement> & {
		open: boolean;
		onClose: () => void;
		children?: Snippet;
	} = $props();

	let dialogEl = $state<HTMLDialogElement | null>(null);

	// Sync the native modal state to the parent-owned `open`. showModal() throws if already open and
	// close() is a no-op when closed, so both transitions are guarded on the element's own `.open`.
	$effect(() => {
		const d = dialogEl;
		if (!d) return;
		if (open && !d.open) d.showModal();
		else if (!open && d.open) d.close();
	});

	function onCancel(e: Event): void {
		// Escape fires `cancel`; keep the element open and route through the parent instead, so `open`
		// stays the single source of truth (the effect then closes the element).
		e.preventDefault();
		onClose();
	}
	function onDialogClick(e: MouseEvent): void {
		// A click whose target is the dialog element itself (not the body/content) is a CANDIDATE
		// backdrop click — but the dialog's own scrollbar is targeted the same way (it belongs to the
		// scroller, not to a child), so target alone would dismiss on a scrollbar grab. Confirm with
		// the coordinates: only a click landing outside the border box is really the backdrop.
		if (e.target !== dialogEl || !dialogEl) return;
		const r = dialogEl.getBoundingClientRect();
		const inside =
			e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
		if (!inside) onClose();
	}
</script>

<dialog
	{...rest}
	bind:this={dialogEl}
	class={`ui-dialog ${klass}`.trim()}
	oncancel={onCancel}
	onclick={onDialogClick}
>
	<div class="ui-dialog-body">
		{@render children?.()}
	</div>
</dialog>

<style>
	.ui-dialog {
		/* No padding so a backdrop click (target === the dialog) is unambiguous — the body owns the
		   inner padding. Centered by the UA's modal margin:auto. */
		padding: 0;
		border: var(--dialog-border, 1px solid var(--border-strong));
		border-radius: var(--dialog-radius, var(--radius-md));
		background: var(--dialog-bg, var(--surface-2));
		color: var(--text);
		box-shadow: var(--dialog-shadow, var(--shadow-2));
		max-width: var(--dialog-max-width, min(90vw, 32rem));
		max-height: var(--dialog-max-height, 85vh);
		overflow: auto;
	}
	.ui-dialog::backdrop {
		/* --bg inherits from :root, so it resolves in the backdrop pseudo across browsers. */
		background: var(--dialog-scrim, color-mix(in srgb, var(--bg) 60%, transparent));
	}
	.ui-dialog-body {
		padding: var(--dialog-pad, var(--space-8));
		min-width: 0;
	}
</style>
