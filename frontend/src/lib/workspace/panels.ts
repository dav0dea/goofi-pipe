/**
 * Built-in panel-type registration.
 *
 * Import this module once at app startup (before the workspace renders) to
 * populate the registry. This is the single place that maps panel-type ids to
 * their content components — adding a panel is a one-line `registerPanel`
 * here, or a `registerPanel` call from any future mod/plugin.
 */
import { registerPanel } from './registry';
import EmptyPanel from './EmptyPanel.svelte';

let done = false;

export function registerBuiltinPanels(): void {
	if (done) return;
	done = true;

	registerPanel({
		id: 'empty',
		title: 'Empty',
		icon: '▢',
		component: EmptyPanel
	});

	// Application panels (node-editor, parameters, viewer, metadata, console)
	// are registered in lib/panels — see registerAppPanels().
}
