<!--
  The node a panel is bound to, as a dropdown — the second door onto a binding drag-and-drop was the
  only way into. Built once and worn by all four panel types that bind a node (parameters, viewer,
  metadata, console), because a viewer on a layout page with no editor on it — or on a phone, where
  there is no drag from another page at all — has to be able to say which node it is watching.

  It is the same `Select` at `density="chrome"` every other panel-bar dropdown is; what it commits is
  the node's UID (the binding's identity), what it shows is the display name, and `nodeChoices.ts`
  owns the mapping between them.
-->
<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import { workspace } from 'tatami';
	import { linkedNodeName } from 'tatami';
	import { Select } from '$lib/ui';
	import { NO_NODE, nodePickList } from './nodeChoices';

	let {
		panelId,
		state,
		emptyLabel
	}: {
		panelId: string;
		/** The panel's opaque state bag — the binding is read out of it, never passed in resolved. */
		state: unknown;
		/** What "nothing bound" says here: a console FILTERS to a node, the others BIND one. */
		emptyLabel: string;
	} = $props();

	const g = graph();
	const ws = workspace();
	// `g.nodes` is the live list, so a node added, renamed or removed anywhere in the patch is in
	// this dropdown on the same tick the canvas draws it.
	const list = $derived(nodePickList(g.nodes, linkedNodeName(state), emptyLabel));

	function choose(uid: string): void {
		if (uid === NO_NODE) ws.unlinkNodeFromPanel(panelId);
		else ws.linkNodeToPanel(panelId, uid);
	}
</script>

<!-- Mono, and inline rather than in a scoped rule: a class in THIS component's <style> never
     reaches another component's markup (the `.pf-identity-bar` lesson), and the node's display name
     is the identifier the canvas paints on the node itself, so it reads in mono wherever it appears
     (D-T3). app.css gives every <select> `font: inherit`, so the face travels down from the
     wrapper this lands on. -->
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
