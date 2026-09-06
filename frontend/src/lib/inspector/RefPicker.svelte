<!--
  RefPicker — a param's reference as two lists, node then slot, each over the live catalogue
  filtered by what this param may reference. The pair commits as one `node.slot`.
-->
<script lang="ts">
	import { Combobox } from '$lib/ui';
	import { liveCatalogue } from './expr/catalogue';
	import { refNodes, refSlots, splitReference, wantedDtype } from './expr/refs';

	let {
		value,
		paramType,
		onCommit,
		testid
	}: {
		/** The committed `node.slot`, or null. */
		value: string | null;
		paramType: string;
		onCommit: (reference: string) => void;
		testid: string;
	} = $props();

	const want = $derived(wantedDtype(paramType));
	let node = $state('');
	let slot = $state('');
	// The committed value is adopted whenever it moves; a half-picked pair stays local until then.
	$effect(() => {
		[node, slot] = splitReference(value);
	});

	function pickNode(n: string): void {
		node = n;
		const slots = refSlots(liveCatalogue(), n, want);
		slot = slots.length === 1 ? slots[0].label : '';
		if (node && slot) onCommit(`${node}.${slot}`);
	}
	function pickSlot(s: string): void {
		slot = s;
		if (node && slot) onCommit(`${node}.${slot}`);
	}
</script>

<div class="ref-picker" data-testid={testid}>
	<Combobox
		value={node}
		options={() => refNodes(liveCatalogue(), want)}
		onCommit={pickNode}
		placeholder="node"
		testid={`${testid}-node`}
	/>
	<span class="dot" aria-hidden="true">.</span>
	<Combobox
		value={slot}
		options={() => refSlots(liveCatalogue(), node, want)}
		onCommit={pickSlot}
		placeholder="slot"
		testid={`${testid}-slot`}
		disabled={!node}
	/>
</div>

<style>
	.ref-picker {
		flex: 1;
		min-width: 0;
		display: flex;
		align-items: stretch;
		gap: var(--space-1);
	}
	.dot {
		align-self: center;
		font-family: var(--font-mono);
		color: var(--text-muted);
	}
</style>
