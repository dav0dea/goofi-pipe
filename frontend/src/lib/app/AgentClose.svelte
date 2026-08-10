<!--
  The detach-or-kill question, asked once for the whole app.

  It is about an INSTANCE, not a view, so it does not belong to a panel. While it did, the TopBar's
  badge — the one door onto a harness from outside its panel — had to OPEN a panel to hold it, and
  with none open that meant `addTab('agent')`: `session_add_page` + `page_set_panel`, two authoring
  commands, so asking a question dirtied the patch and pushed two undo steps. Navigation must not
  dirty (CLAUDE.md), and asking is not even navigation. As shell chrome the question needs no layout
  write at all, and both doors — the panel header's ✕ and the badge — land on this one dialog.

  `hs.closing` is the whole state: which instance was asked about, and the panel to close once it is
  answered (the header's ✕ closes; the badge does not).
-->
<script lang="ts">
	import { harnesses, harnessLabel } from '$lib/stores/harness.svelte';
	import { detachTermSession } from '$lib/stores/termSession';
	import { workspace } from '$lib/workspace/workspace.svelte';
	import { Button, Dialog } from '$lib/ui';

	const hs = harnesses();
	const ws = workspace();

	const inst = $derived(hs.instances.find((i) => i.id === hs.closing?.id) ?? null);
	/** A dead child cannot be left running, so the same two doors mean different things for it:
	 * letting go keeps its last screen on the roster, and the manager's stop DISMISSES it. */
	const exited = $derived(inst?.state === 'exited');
	const name = $derived(inst ? harnessLabel(inst) : 'the harness');

	function answer(kill: boolean): void {
		const at = hs.closing;
		if (!at) return;
		if (kill) {
			hs.kill(at.id);
		} else {
			detachTermSession(at.id);
			const panel = hs.panelShowing(at.id);
			if (panel) hs.release(panel);
			else hs.cancelClose();
		}
		if (at.closePanel) ws.close(at.closePanel);
	}
</script>

<Dialog open={!!hs.closing} onClose={() => hs.cancelClose()} data-testid="agent-close-dialog">
	<h2>{exited ? 'Dismiss this agent?' : 'Close this agent view?'}</h2>
	<p>
		{#if exited}
			{name} has already exited. Hiding it keeps its last screen — re-open it from any agent panel —
			and dismissing drops it from the roster for good.
		{:else}
			Detaching leaves {name} running in the patch workspace — re-attach it from any agent panel. Killing
			stops it.
		{/if}
	</p>
	<div class="choices">
		<Button data-testid="agent-detach" onclick={() => answer(false)}>
			{exited ? 'Hide' : 'Detach'}
		</Button>
		<Button variant="danger" data-testid="agent-kill" onclick={() => answer(true)}>
			{exited ? 'Dismiss' : 'Kill'}
		</Button>
		<Button variant="ghost" onclick={() => hs.cancelClose()}>Cancel</Button>
	</div>
</Dialog>

<style>
	/* A dialog's actions sit at its END — the launcher's choices centre under their prompt because
	   they ARE the empty state's content; these answer a question the sentence above just asked. */
	.choices {
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-3);
		justify-content: flex-end;
		margin-top: var(--space-6);
	}
	h2 {
		margin: 0 0 var(--space-4);
		font-size: var(--fs-body);
		font-weight: 600;
	}
	p {
		margin: 0;
		color: var(--text-dim);
		font-size: var(--fs-small);
	}
</style>
