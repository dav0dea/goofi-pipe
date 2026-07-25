<!--
  Transient toast for undo/redo failures (#9). Watches history().lastError — set
  when an undo/redo replay rejects (e.g. undo-of-delete onto a re-taken name) — shows
  it for a few seconds, then clears it. Clicking dismisses immediately. Sits above all
  other chrome (var(--z-toast)).
-->
<script lang="ts">
	import { history } from '$lib/stores/history.svelte';

	const error = $derived(history().lastError);

	$effect(() => {
		if (!error) return;
		const t = setTimeout(() => history().clearError(), 4000);
		return () => clearTimeout(t);
	});
</script>

{#if error}
	<button class="toast" onclick={() => history().clearError()} title="Dismiss">
		{error}
	</button>
{/if}

<style>
	/* Kept bespoke on purpose: the button IS the toast surface (fixed, centred, self-styled), which
	   no Button variant expresses. `font: inherit` is its own — the base rule keeps only that reset
	   today and M-Task 7 strips the rest of the skin. */
	.toast {
		position: fixed;
		bottom: var(--space-8);
		left: 50%;
		transform: translateX(-50%);
		z-index: var(--z-toast);
		max-width: 60ch;
		padding: var(--space-4) var(--space-7);
		border: none;
		border-radius: var(--radius-md);
		background: var(--danger);
		color: var(--on-danger);
		font: inherit;
		font-size: var(--fs-small);
		text-align: left;
		box-shadow: var(--shadow-1);
		cursor: pointer;
	}
</style>
