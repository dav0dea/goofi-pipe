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
		acceptsNode: true,
		// Rings its own content via NodeLinkedPanel (as do Viewer and Metadata).
		contentOutline: true
	});
	registerPanel({
		id: 'viewer',
		title: 'Viewer',
		icon: '◫',
		component: ViewerPanel,
		acceptsNode: true,
		contentOutline: true
	});
	registerPanel({
		id: 'metadata',
		title: 'Metadata',
		icon: 'ⓘ',
		component: MetadataInspectorPanel,
		acceptsNode: true,
		contentOutline: true
	});
	registerPanel({
		id: 'console',
		title: 'Console',
		icon: '▤',
		component: ConsolePanel,
		// Dropping a node filters the console to just that node's output. No `contentOutline`:
		// it draws no ring of its own, so it takes the panel frame ring like the node editor.
		acceptsNode: true
	});
	// Patch globals (default_ufreq + user-defined scalars). Not in the default layout —
	// opened on demand from the panel-type menu (like a secondary inspector).
	registerPanel({
		id: 'globals',
		title: 'Globals',
		icon: '⧉',
		component: GlobalsPanel,
		// Not a node-drop target, but rings its own content (below the header bar) like the
		// node-linked panels do, so it opts out of the panel frame ring the same way.
		contentOutline: true
	});
	// The old dockable "Errors" panel was removed — the Console (filterable,
	// accumulating, stderr-aware) supersedes it, and a legacy `errors` panel type
	// migrates to `console` on load (see workspace.svelte.ts). Per-node current
	// errors still surface on the floating chip and the inspector's error section.
}
