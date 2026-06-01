/**
 * Application panel registration. Maps the app's panel-type ids to their
 * content components. Called once at startup (AppShell). Mods can register
 * further panel types the same way via `registerPanel`.
 */
import { registerPanel } from '$lib/workspace/registry';
import NodeEditorPanel from './NodeEditorPanel.svelte';

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
}
