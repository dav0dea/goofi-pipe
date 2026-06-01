/**
 * Application panel registration. Maps the app's panel-type ids to their
 * content components. Called once at startup (AppShell). Mods can register
 * further panel types the same way via `registerPanel`.
 */
import { registerPanel } from '$lib/workspace/registry';
import NodeEditorPanel from './NodeEditorPanel.svelte';
import ParametersPanel from './ParametersPanel.svelte';
import ViewerPanel from './ViewerPanel.svelte';
import MetadataInspectorPanel from './MetadataInspectorPanel.svelte';
import ErrorsPanel from './ErrorsPanel.svelte';

let done = false;

export function registerAppPanels(): void {
	if (done) return;
	done = true;

	registerPanel({
		id: 'node-editor',
		title: 'Node Editor',
		icon: '⛓',
		component: NodeEditorPanel
	});
	registerPanel({
		id: 'parameters',
		title: 'Parameters',
		icon: '☰',
		component: ParametersPanel
	});
	registerPanel({
		id: 'viewer',
		title: 'Viewer',
		icon: '◫',
		component: ViewerPanel
	});
	registerPanel({
		id: 'metadata',
		title: 'Metadata',
		icon: 'ⓘ',
		component: MetadataInspectorPanel
	});
	registerPanel({
		id: 'errors',
		title: 'Errors',
		icon: '⚠',
		component: ErrorsPanel
	});
}
