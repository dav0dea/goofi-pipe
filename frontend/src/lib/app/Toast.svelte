<!--
  The app's transient alarm surface. Watches the shared `notify` channel — an undo/redo replay that
  rejected (#9), a save or load that failed — shows it for a few seconds, then clears it. Clicking
  dismisses immediately. Sits above all other chrome (var(--z-toast)).

  It reads ONE store on purpose: the channel is where the two producers meet, so this component
  never learns who raised the line it is showing.
-->
<script lang="ts">
	import { notify } from '$lib/stores/notify.svelte';

	// The singleton is reached HERE, above the derived. `notify()` constructs it lazily and this
	// component is its first caller, so calling it INSIDE the derived created the store's `$state`
	// in that tracking scope — where every later `raise()` set the message and re-rendered nothing.
	// Both producers were silent for it. `stores/singletonScope.test.ts` is the guard.
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
