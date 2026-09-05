<script lang="ts">
	import { demoIdle } from '$lib/stores/demoIdle.svelte';
	import { Button, Dialog } from '$lib/ui';

	const idle = demoIdle();

	// Dismissing without answering is not staying: the manager decides, and silence is the answer
	// it already read. The dialog closes, and the sockets close on their own schedule.
	const left = $derived(idle.secondsLeft);
</script>

<Dialog open={idle.pending} onClose={() => idle.stay()} data-testid="demo-idle-dialog">
	<h2>Still there?</h2>
	<p>
		This public demo has been untouched for a while. It hands itself back in
		<strong data-testid="demo-idle-seconds">{left}s</strong> so it stops costing its host — the
		patch is shared, so whatever is on the canvas goes with it.
	</p>
	<div class="choices">
		<Button data-testid="demo-idle-stay" onclick={() => idle.stay()}>Keep it open</Button>
	</div>
</Dialog>

<style>
	.choices {
		display: flex;
		gap: var(--space-3);
		justify-content: flex-end;
		margin-top: var(--space-6);
	}
</style>
