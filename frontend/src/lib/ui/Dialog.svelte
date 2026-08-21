<!-- Dialog — a centered modal on the native `<dialog>`; the parent owns `open`, and every
     dismissal routes to `onClose` rather than closing the element behind the parent's back. -->
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

	// Both transitions are guarded on `.open`: showModal() throws when the dialog is already open.
	$effect(() => {
		const d = dialogEl;
		if (!d) return;
		if (open && !d.open) d.showModal();
		else if (!open && d.open) d.close();
	});

	function onCancel(e: Event): void {
		// Keep the element open and route through the parent, which owns `open`.
		e.preventDefault();
		onClose();
	}
	function onDialogClick(e: MouseEvent): void {
		// The dialog's own scrollbar is targeted like the backdrop, so the coordinates decide.
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
		/* No padding, so a backdrop click is unambiguous; the body owns the inner padding. */
		padding: 0;
		border: var(--dialog-border, 1px solid var(--border-strong));
		border-radius: var(--dialog-radius, var(--radius-md));
		background: var(--dialog-bg, var(--surface-2));
		color: var(--text);
		box-shadow: var(--dialog-shadow, var(--shadow-2));
		max-width: var(--dialog-max-width, min(90vw, 32rem));
		max-height: var(--dialog-max-height, 85dvh);
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
