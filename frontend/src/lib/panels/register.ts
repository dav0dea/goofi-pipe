import type { Component } from 'svelte';
import { registerPanel, type PanelProps } from 'panelty';
import { PANEL_TYPES, type PanelTypeId } from '$lib/api/vocab';
import { harnesses } from '$lib/stores/harness.svelte';
import EmptyPanel from './EmptyPanel.svelte';
import NodeEditorPanel from './NodeEditorPanel.svelte';
import ParametersPanel from './ParametersPanel.svelte';
import ViewerPanel from './ViewerPanel.svelte';
import MetadataInspectorPanel from './MetadataInspectorPanel.svelte';
import ConsolePanel from './ConsolePanel.svelte';
import GlobalsPanel from './GlobalsPanel.svelte';
import AgentPanel from './AgentPanel.svelte';

const components: Record<PanelTypeId, Component<PanelProps>> = {
	empty: EmptyPanel,
	'node-editor': NodeEditorPanel,
	parameters: ParametersPanel,
	viewer: ViewerPanel,
	metadata: MetadataInspectorPanel,
	console: ConsolePanel,
	globals: GlobalsPanel,
	agent: AgentPanel
};

/** Panel types that answer their own ✕: closing an agent view must not silently kill the agent. */
const confirmClose: Partial<Record<PanelTypeId, (panelId: string) => boolean>> = {
	agent: (panelId) => {
		const id = harnesses().instanceFor(panelId);
		if (!id) return false;
		harnesses().requestClose(id, panelId);
		return true;
	}
};

let done = false;

export function registerAppPanels(): void {
	if (done) return;
	done = true;

	for (const t of PANEL_TYPES) {
		registerPanel({
			id: t.id,
			title: t.title,
			icon: t.icon,
			component: components[t.id],
			acceptsNode: t.acceptsNode,
			confirmClose: confirmClose[t.id]
		});
	}
}
