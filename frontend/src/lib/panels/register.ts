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
import ErrorsPanel from './ErrorsPanel.svelte';

let done = false;

export function registerAppPanels(): void {
	if (done) return;
	done = true;

	// The trailing ︎ (text-presentation variation selector) forces the two
	// emoji-default glyphs (⛓ U+26D3, ⚠ U+26A0) to render as monochrome text so
	// they honor the inherited icon color — without it the browser draws them from
	// the colour-emoji font, leaving these two coloured while the rest are grey.
	registerPanel({
		id: 'node-editor',
		title: 'Node Editor',
		icon: '⛓︎',
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
	registerPanel({
		id: 'errors',
		title: 'Errors',
		icon: '⚠︎',
		component: ErrorsPanel
	});
}
