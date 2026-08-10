/**
 * Built-in panel-type registration.
 *
 * Import this module once at app startup (before the workspace renders) to
 * populate the registry with the framework's OWN placeholder. Every application
 * panel is registered one layer up, in lib/panels — see registerAppPanels().
 * Both read the same vocabulary, which the manager declares.
 */
import { registerPanel } from './registry';
import { PANEL_TYPES, EMPTY_PANEL_TYPE } from '$lib/api/vocab';
import EmptyPanel from './EmptyPanel.svelte';

let done = false;

export function registerBuiltinPanels(): void {
	if (done) return;
	done = true;

	// First, so it leads the panel menu — and so a fresh split's placeholder is renderable before
	// any application panel exists.
	for (const t of PANEL_TYPES.filter((t) => t.id === EMPTY_PANEL_TYPE)) {
		registerPanel({ id: t.id, title: t.title, icon: t.icon, component: EmptyPanel });
	}
}
