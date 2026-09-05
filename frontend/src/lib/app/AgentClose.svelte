<script lang="ts">
	import { harnesses, harnessLabel } from '$lib/stores/harness.svelte';
	import { detachTermSession } from '$lib/stores/termSession';
	import { workspace } from 'panelty';
	import { Button, Dialog } from '$lib/ui';

	const hs = harnesses();
	const ws = workspace();

	const inst = $derived(hs.instances.find((i) => i.id === hs.closing?.id) ?? null);
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
	<h2>Close this agent view?</h2>
	<p>
		Detaching leaves {name} running in the patch workspace — re-attach it from any agent panel. Killing
		stops it.
	</p>
	<div class="choices">
		<Button data-testid="agent-detach" onclick={() => answer(false)}>Detach</Button>
		<Button variant="danger" data-testid="agent-kill" onclick={() => answer(true)}>Kill</Button>
		<Button variant="ghost" onclick={() => hs.cancelClose()}>Cancel</Button>
	</div>
</Dialog>

<style>
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
