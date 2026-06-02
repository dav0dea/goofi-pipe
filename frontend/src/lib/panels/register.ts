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
import ConsolePanel from './ConsolePanel.svelte';

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
		component: ParametersPanel,
		acceptsNode: true
	});
	registerPanel({
		id: 'viewer',
		title: 'Viewer',
		icon: '◫',
		component: ViewerPanel,
		acceptsNode: true
	});
	registerPanel({
		id: 'metadata',
		title: 'Metadata',
		icon: 'ⓘ',
		component: MetadataInspectorPanel,
		acceptsNode: true
	});
	registerPanel({
		id: 'console',
		title: 'Console',
		icon: '▤',
		component: ConsolePanel,
		// Dropping a node filters the console to just that node's output.
		acceptsNode: true
	});
}
