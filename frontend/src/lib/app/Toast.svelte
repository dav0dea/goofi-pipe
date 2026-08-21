<script lang="ts">
	import { notify } from '$lib/stores/notify.svelte';

	// Above the derived: this is the store's first caller, and constructing its `$state` inside a
	// tracking scope makes every later `raise()` silent.
	const n = notify();
	const error = $derived(n.message);

	$effect(() => {
		if (!error) return;
		const t = setTimeout(() => n.clear(), 4000);
		return () => clearTimeout(t);
	});
</script>

{#if error}
	<button class="toast" data-testid="toast" onclick={() => n.clear()} title="Dismiss">
		{error}
	</button>
{/if}

<style>
	.toast {
		position: fixed;
		/* The shell's safe-area padding cannot reach a fixed element, so the insets are restated. */
		bottom: calc(var(--space-8) + var(--kb-inset, 0px) + var(--safe-bottom, 0px));
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
