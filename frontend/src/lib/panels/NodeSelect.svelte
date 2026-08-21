<!-- The node a panel is bound to, as a dropdown: it commits the uid and shows the display name. -->
<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import { workspace } from 'panelty';
	import { linkedNodeName } from 'panelty';
	import { Select } from '$lib/ui';
	import { NO_NODE, nodePickList } from './nodeChoices';

	let {
		panelId,
		state,
		emptyLabel
	}: {
		panelId: string;
		state: unknown;
		/** What "nothing bound" says here: a console FILTERS to a node, the others BIND one. */
		emptyLabel: string;
	} = $props();

	const g = graph();
	const ws = workspace();
	const list = $derived(nodePickList(g.nodes, linkedNodeName(state), emptyLabel));

	function choose(uid: string): void {
		if (uid === NO_NODE) ws.unlinkNodeFromPanel(panelId);
		else ws.linkNodeToPanel(panelId, uid);
	}
</script>

<!-- Mono inline, not in a scoped rule: a class here never reaches another component's markup. -->
<Select
	density="chrome"
	style="font-family: var(--font-mono)"
	value={list.value}
	options={list.options}
	labels={list.labels}
	onChange={choose}
	title="Bound node"
	data-testid="panel-node"
/>
