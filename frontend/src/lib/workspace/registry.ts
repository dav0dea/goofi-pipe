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

/** The single prop contract every panel content component receives. */
export interface PanelProps {
	/** Stable id of the hosting panel (for keying subscriptions, etc.). */
	panelId: string;
	/** Opaque, persisted per-panel state — `undefined` until the panel sets it. */
	state: unknown;
	/** Persist new per-panel state back into the layout tree. */
	setState: (s: unknown) => void;
	/** True when this is the active (last-focused) panel — used to scope
	 * keyboard shortcuts so only the focused panel reacts. */
	active: boolean;
}

export interface PanelType {
	/** Registry key, stored in the layout as `PanelNode.panelType`. */
	id: string;
	/** Human label shown in the header dropdown and context menu. */
	title: string;
	/** Optional glyph shown beside the title. */
	icon?: string;
	component: Component<PanelProps>;
	/** Factory for the panel's initial state when first created / switched to. */
	defaultState?: () => unknown;
	/** When false, the panel renders its own header chrome instead of the
	 * framework's. Defaults to true. */
	chrome?: boolean;
	/** True if a node dragged from an editor can be dropped onto this panel to
	 * bind it (Parameters / Viewer / Metadata). */
	acceptsNode?: boolean;
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
