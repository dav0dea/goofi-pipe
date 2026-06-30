<!-- Viewer panel — visualizes an output slot of the node dragged into it. The
     slot picker plus the shared viewer-type dropdown + settings cog live in the
     panel header (next to the node name); the body is the shared ViewerFeed.
     Each panel is an INDEPENDENT viewer instance: its viewer type + settings
     live in this panel's own layout state (a panelBinding) and persist with the
     panel, so it never tracks the node's inline viewer or another panel. The
     data stream is still shared (one WS per node+slot). Linking + empty state
     via NodeLinkedPanel. -->
<script lang="ts">
	import type { PanelProps } from '$lib/workspace/registry';
	import type { NodeInstanceInfo } from '$lib/api/control';
	import NodeLinkedPanel from './NodeLinkedPanel.svelte';
	import ViewerFeed from '$lib/viewers/ViewerFeed.svelte';
	import ViewerControls from '$lib/viewers/ViewerControls.svelte';
	import { panelBinding, type ViewBinding } from '$lib/viewers/viewBinding';
	import { recordViewChange } from '$lib/viewers/viewExecutors';
	import { asStateObject } from '$lib/workspace/panelState';
	import type { ViewerKind } from '$lib/viewers/kind';
	import type { SettingsMap } from '$lib/viewers/settingsSchema';

	interface ViewerState {
		node?: string | null;
		slot?: string | null;
	}

	let props: PanelProps = $props();

	function st(): ViewerState {
		return asStateObject(props.state) as ViewerState;
	}
	function curSlot(node: NodeInstanceInfo): string | null {
		const cur = st();
		const names = Object.keys(node.output_slots);
		return cur.slot && node.output_slots[cur.slot] ? cur.slot : (names[0] ?? null);
	}
	function pick(slot: string): void {
		props.setState({ ...st(), slot });
	}

	// Resolve the chosen slot, its dtype, and this panel's binding in one place so
	// the controls and content snippets (separate scopes) don't each re-derive it.
	// Raw (pre-resolution) snapshot of this panel's view state, for undo capture.
	function snap(): { kind?: ViewerKind; settings: SettingsMap } {
		const s = asStateObject(props.state);
		return { kind: s.kind as ViewerKind | undefined, settings: (s.settings as SettingsMap) ?? {} };
	}
	function view(node: NodeInstanceInfo): { slot: string | null; dtype: string | null; binding: ViewBinding } {
		const slot = curSlot(node);
		const dtype = slot ? node.output_slots[slot] : null;
		const inner = panelBinding(() => props.state, props.setState, dtype);
		// Wrap the setters so a viewer-type/settings change records an undo step
		// (the inner binding stays pure + unit-testable in viewBinding.ts).
		const binding: ViewBinding = {
			get kind() {
				return inner.kind;
			},
			get settings() {
				return inner.settings;
			},
			setKind(k) {
				const before = snap();
				inner.setKind(k);
				recordViewChange({ kind: 'panel', panelId: props.panelId }, before, snap(), `Viewer → ${k}`);
			},
			setSetting(key, value) {
				const before = snap();
				inner.setSetting(key, value);
				recordViewChange({ kind: 'panel', panelId: props.panelId }, before, snap(), `Viewer ${key}`);
			}
		};
		return { slot, dtype, binding };
	}
</script>

<NodeLinkedPanel {...props} label="data">
	{#snippet controls(node)}
		{@const { slot, dtype, binding } = view(node)}
		<select
			class="slot-pick"
			value={slot ?? ''}
			onchange={(e) => pick(e.currentTarget.value)}
			data-testid="viewer-slot"
		>
			{#each Object.entries(node.output_slots) as [name, dt] (name)}
				<option value={name}>{node.slot_labels?.[name] ?? name} · {dt.toLowerCase()}</option>
			{/each}
		</select>
		{#if slot && dtype}
			<ViewerControls {dtype} {binding} />
		{/if}
	{/snippet}

	{#snippet content(node)}
		{@const { slot, binding } = view(node)}
		<div class="vp-body"><ViewerFeed node={node.name} {slot} {binding} /></div>
	{/snippet}
</NodeLinkedPanel>

<style>
	.slot-pick {
		flex: 0 1 auto;
		min-width: 0;
		font-size: 0.78rem;
		padding: 2px 6px;
	}
	.vp-body {
		flex: 1;
		min-height: 0;
		display: flex;
		padding: 6px;
	}
</style>
