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
import GlobalsPanel from './GlobalsPanel.svelte';

let done = false;

export function registerAppPanels(): void {
	if (done) return;
	done = true;

	// Panel icons are text-presentation glyphs (Geometric Shapes / Misc Technical),
	// NOT emoji — so they all render monochrome and honor the inherited grey.
	// Avoid emoji-default codepoints (⛓ U+26D3, ⚠ U+26A0) here: the browser draws
	// those from the colour-emoji font regardless of CSS.
	registerPanel({
		id: 'node-editor',
		title: 'Node Editor',
		icon: '◈',
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
	// Patch globals (default_ufreq + user-defined scalars). Not in the default layout —
	// opened on demand from the panel-type menu (like a secondary inspector).
	registerPanel({
		id: 'globals',
		title: 'Globals',
		icon: '⧉',
		component: GlobalsPanel
	});
	// The old dockable "Errors" panel was removed — the Console (filterable,
	// accumulating, stderr-aware) supersedes it, and a legacy `errors` panel type
	// migrates to `console` on load (see workspace.svelte.ts). Per-node current
	// errors still surface on the floating chip and the inspector's error section.
}
