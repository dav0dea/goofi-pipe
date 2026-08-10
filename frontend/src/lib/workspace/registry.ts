/**
 * Panel-type registry — the moddability seam.
 *
 * The panel framework knows nothing about any specific panel; it only knows
 * how to render whatever components are registered here. To add a new panel
 * type (now or as a future mod) you write a Svelte component honoring
 * `PanelProps` and call `registerPanel({...})` once at startup. The content
 * dropdown, context menu, and layout persistence all pick it up automatically.
 */
import type { Component } from 'svelte';
import type { IconName } from '$lib/ui';
import type { LayoutIntent } from './workspace.svelte';

/** The single prop contract every panel content component receives. */
export interface PanelProps {
	/** Stable id of the hosting panel (for keying subscriptions, etc.). */
	panelId: string;
	/** Opaque, persisted per-panel state — `undefined` until the panel sets it. */
	state: unknown;
	/** Persist new per-panel state. `intent` routes it, and the routing IS the dirty taxonomy: pass
	 * `'navigation'` when the user only changed what the panel is LOOKING at (entering a sub-patch)
	 * and it stays this client's viewpoint; omit it for an edit, which becomes one `page_set_panel`
	 * command — undoable, converged to peers, and dirtying. `label` names that undo step. */
	setState: (s: unknown, intent?: LayoutIntent, label?: string) => void;
}

export interface PanelType {
	/** Registry key, stored in the layout as `PanelNode.panelType`. */
	id: string;
	/** Human label shown in the header dropdown and context menu. */
	title: string;
	/** Optional icon shown beside the title, named from the app's one icon set. A mod picks from
	 * that set rather than shipping its own artwork — which is what keeps the panel menu one
	 * visual system however many panel types are registered into it. */
	icon?: IconName;
	component: Component<PanelProps>;
	/** True if a node dragged from an editor can be dropped onto this panel to
	 * bind it (Parameters / Viewer / Metadata). */
	acceptsNode?: boolean;
	/** A panel type's chance to answer its own ✕. Return true to say the close has been TAKEN
	 * OVER — the panel raised its own question and will do the closing itself, or not at all.
	 * Exists because closing an agent view is not the same act as killing the long-running agent
	 * behind it, and the header must not decide that on the user's behalf. */
	confirmClose?: (panelId: string) => boolean;
}

const registry = new Map<string, PanelType>();
/** Insertion order, so the dropdown lists panels in registration order. */
const order: string[] = [];

export function registerPanel(type: PanelType): void {
	if (!registry.has(type.id)) order.push(type.id);
	registry.set(type.id, type);
}

export function getPanelType(id: string): PanelType | undefined {
	return registry.get(id);
}

export function listPanelTypes(): PanelType[] {
	return order.map((id) => registry.get(id)).filter((t): t is PanelType => t !== undefined);
}

/** Resolve a panel type, falling back to a synthetic "unknown" descriptor so a
 * layout referencing an unregistered type still renders something sane. */
export function resolvePanelType(id: string): PanelType {
	const t = registry.get(id);
	if (t) return t;
	return { id, title: id, component: undefined as unknown as Component<PanelProps> };
}
