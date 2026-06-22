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
	.toast {
		position: fixed;
		bottom: 18px;
		left: 50%;
		transform: translateX(-50%);
		z-index: var(--z-toast);
		max-width: 60ch;
		padding: 8px 14px;
		border: none;
		border-radius: 6px;
		background: var(--danger);
		color: #fff;
		font: inherit;
		font-size: 0.85rem;
		text-align: left;
		box-shadow: 0 6px 20px rgb(0 0 0 / 0.35);
		cursor: pointer;
	}
</style>
