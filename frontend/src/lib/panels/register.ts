/**
 * Application panel registration — the id→component half of a panel type.
 *
 * The vocabulary (which types exist, their titles, icons and whether each binds a
 * node) is the manager's: declared in `backend/goofi-bridge/src/vocab.rs` and
 * generated into `$lib/api/vocab`, so a panel type the manager will not accept
 * cannot be registered here and one it mints cannot be missing. What stays here
 * is the component, which is code and could not live anywhere else. Called once
 * at startup (AppShell). Mods can register further panel types the same way via
 * `registerPanel`.
 */
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

/** Every app panel type, exhaustively: a type added to the manager's table is a `npm run check`
 * error here until it has something to draw it. `empty` included — the panel framework MINTS that
 * type on a split and draws whatever is registered for it, and what a blank panel offers is a
 * question about this app's vocabulary. */
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

/** The panel types that answer their own ✕ (see `PanelType.confirmClose`). Closing an agent view
 * must not silently kill a long-running agent, so the panel raises the detach-or-kill question
 * instead — and only when there is a live instance in it to ask about. */
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

	// Panel icons are names from the app's one icon set ($lib/ui/icons), so they are drawn by the
	// app at the app's weight. They used to be text-presentation glyphs picked to dodge the
	// colour-emoji font (⛓ U+26D3 and ⚠ U+26A0 are emoji-default, and the browser draws those in
	// colour regardless of CSS) — a constraint the icon set removes rather than works around.
	//
	// Registration order is the table's order, which is the order the panel menu and the empty
	// panel's choice grid list them — `empty` leads it, so a fresh split's placeholder is drawable
	// before anything else is. Globals is not in the default layout — it is opened on demand from
	// that menu, like a secondary inspector.
	//
	// The old dockable "Errors" panel was removed — the Console (filterable, accumulating,
	// stderr-aware) supersedes it, and a legacy `errors` panel type migrates to `console` on load
	// (see workspace.svelte.ts). Per-node current errors still surface on the floating chip and the
	// inspector's error section.
	for (const t of PANEL_TYPES) {
		registerPanel({
			id: t.id,
			title: t.title,
			icon: t.icon,
			component: components[t.id],
			// Dropping a node onto a Parameters/Viewer/Metadata panel binds it; onto the Console it
			// filters the log to that node.
			acceptsNode: t.acceptsNode,
			confirmClose: confirmClose[t.id]
		});
	}
}
