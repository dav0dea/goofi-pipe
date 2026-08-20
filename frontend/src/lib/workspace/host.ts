/**
 * The seam between the panel system and whatever owns the arrangement.
 *
 * The panel system holds no tree. It draws the one it is handed (`syncFromDoc`), recognises the
 * gestures over it, and raises each one HERE — as an intent, not as a write. Whoever implements
 * this owns persistence, concurrency, undo and the naming of tabs; goofi's implementation turns
 * each call into one manager command over `/control`.
 *
 * **Two contracts, each naming only what its own component can trigger.** A consumer rendering the
 * tab strip alone is never handed a method about panels, and a consumer rendering one tab's panels
 * alone is never handed a method about tabs. `tabFromPanel` is optional for the same reason: it is
 * the one gesture that spans both, so it exists only when the two are composed, and a host that
 * cannot express it simply does not offer the drag.
 *
 * Every write answers whether it LANDED. A refusal is ordinary — closing a tab's last panel, a
 * peer having already taken what this gesture names — and the panel system uses the answer to
 * decide where to move the focus, never to retry.
 */
import type { Direction, Workspace } from './model';

/** What a tab op answers: the ids the host minted, which no caller can otherwise know. */
export interface TabRef {
	tab: string;
	panel: string;
}

export interface TabHost {
	/** Add a tab, and answer the ids it minted. The NAME is the host's — a label is not addressing,
	 * and only the host knows what the other tabs are called. `panelType` overrides what its one
	 * panel starts as, which is the agent façade's door onto "a tab showing X". */
	addTab(opts?: { index?: number; panelType?: string }): Promise<TabRef | null>;
	removeTab(tab: string): Promise<boolean>;
	renameTab(tab: string, name: string): Promise<boolean>;
	reorderTab(tab: string, toIndex: number): Promise<boolean>;
}

export interface PanelHost {
	/** Split a panel, and answer the new panel's id — the caller's next act is to give it content,
	 * which needs an id it cannot otherwise know. */
	splitPanel(
		panel: string,
		direction: Direction,
		placeBefore: boolean,
		ratio: number
	): Promise<string | null>;
	removePanel(panel: string): Promise<boolean>;
	/** Every child's share at once, in child order — what a resize drag commits on pointer-up. */
	resizeSplit(split: string, fractions: number[]): Promise<boolean>;
	/** A panel's type and/or its opaque state. `state` MERGES key by key, so a caller sends only
	 * what changed. `label` names the step a host with an undo history records. */
	setPanel(
		panel: string,
		patch: { type?: string; state?: Record<string, unknown> },
		label?: string
	): Promise<boolean>;
	/** Move a subtree beside `target`, splitting it along `direction`. A tab dragged onto a panel
	 * is this too — the panel system resolves it to the tab's root before calling, so there is one
	 * move and not two that must agree. */
	movePanel(
		subtree: string,
		target: string,
		direction: Direction,
		placeBefore: boolean,
		ratio: number
	): Promise<boolean>;
	/** Tear a subtree out into a tab of its own — the drop-onto-the-tab-bar gesture. Present only
	 * when panels are composed inside a tab strip. */
	tabFromPanel?(subtree: string, index: number): Promise<TabRef | null>;
}

/** Both, for the composed case — which is what goofi is. */
export type LayoutHost = TabHost & PanelHost;

/** The tabs a host holds, as the panel system draws them. Pushed in rather than pulled: goofi's
 * arrangement arrives as a document delta, so the push is what already exists and a pull would
 * make every render re-read the document. */
export type TabsIn = (tabs: Workspace[]) => void;
