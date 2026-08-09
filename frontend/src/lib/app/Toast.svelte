<!--
  The app's transient alarm surface. Watches the shared `notify` channel — an undo/redo replay that
  rejected (#9), a save that failed, a workspace whose watch was lost — shows it for a few seconds,
  then clears it. Clicking dismisses immediately. Sits above all other chrome (var(--z-toast)).

  It reads ONE store on purpose: the channel is where the three producers meet, so this component
  never learns who raised the line it is showing.
-->
<script lang="ts">
	import { notify } from '$lib/stores/notify.svelte';

	const error = $derived(notify().message);

	$effect(() => {
		if (!error) return;
		const t = setTimeout(() => notify().clear(), 4000);
		return () => clearTimeout(t);
	});
</script>

{#if error}
	<button class="toast" data-testid="toast" onclick={() => notify().clear()} title="Dismiss">
		{error}
	</button>
{/if}

<style>
	/* Kept bespoke on purpose: the button IS the toast surface (fixed, centred, self-styled), which
	   no Button variant expresses. `font: inherit` is its own — the base rule kept only that reset
	   when M-Task 7 stripped the skin, and a bespoke surface states its whole face regardless. */
	.toast {
		position: fixed;
		/* Lifted clear of the soft keyboard AND of the home indicator / gesture bar: this is the
		   ONLY surface an undo/redo failure is reported on and its only dismissal is a click on
		   itself, so a toast under either is an unreadable, undismissable error. The shell's
		   safe-area padding cannot reach a `position: fixed` element — it is laid out against the
		   initial containing block — so the inset is restated here, through the same token the
		   shell uses. Both terms are 0 on a desktop and whenever the keyboard is down, so the
		   resting geometry is unchanged. */
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
