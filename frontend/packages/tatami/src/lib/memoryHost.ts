/**
 * A [`LayoutHost`] that owns the arrangement itself, in memory.
 *
 * The package holds no tree: it draws what it is handed and raises every gesture through a host.
 * That is the right shape when something else already owns the arrangement — a server, a document,
 * an undo stack — and it is a wall in front of anyone who just wants panels. This is the other end:
 * a host with nowhere to persist to, which plans each gesture over a plain tree and pushes the
 * result straight back in.
 *
 * It is also what proves the port. Every method here is written against `LayoutHost` and nothing
 * else, so a gesture the panel system cannot express through the contract has nowhere to go.
 *
 * What it deliberately does NOT do is undo: an inverse belongs to whoever owns the history, and a
 * host with no history has no honest one to offer. A consumer that wants it implements the same
 * contract over its own command log — which is what goofi does.
 */
import type { Landing, LayoutHost, TabRef } from './host';
import {
	findNode,
	findParent,
	type Direction,
	type LayoutNode,
	type Workspace
} from './model';
import { workspace } from './workspace.svelte';

/** The panel type a fresh tab starts with, and the one a split births. Both are the CONSUMER's
 *  vocabulary — these are the neutral spellings, overridable per call and per host. */
export interface MemoryHostOptions {
	tabs?: Workspace[];
	defaultPanelType?: string;
	emptyPanelType?: string;
	/** Where the plan goes once it is made. Defaults to the panel system's own store. */
	push?: (tabs: Workspace[]) => void;
}

export function memoryHost(opts: MemoryHostOptions = {}): LayoutHost {
	const empty = opts.emptyPanelType ?? 'empty';
	const fresh = opts.defaultPanelType ?? empty;
	const push = opts.push ?? ((t: Workspace[]) => workspace().syncFromDoc(t));

	let tabs: Workspace[] = opts.tabs ? structuredClone(opts.tabs) : [];
	// One counter for every id this host mints, never reused: a viewpoint, a subscription and an
	// undo entry are all keyed by id, so a recycled one is a stale reference that resolves.
	let seq = 0;
	const mint = (kind: string): string => `${kind}-${++seq}`;

	// Seeded with one tab, so a consumer that passes nothing still has something to draw.
	if (tabs.length === 0) tabs = [{ id: mint('tab'), name: 'Tab 1', root: panel(fresh) }];
	else seq = highestSeq(tabs);
	// Handed over at once. The panel system has no `getTabs` to pull with — the arrangement arrives
	// as a push, because for every other host it arrives as one (a delta, a snapshot, a message) and
	// a pull would make every render re-read whatever is behind it.
	commit();

	function panel(panelType: string): LayoutNode {
		return { kind: 'panel', id: mint('panel'), panelType };
	}

	function commit(): true {
		push(structuredClone(tabs));
		return true;
	}

	/** The tab holding `id`, and the tab itself when `id` names one. */
	function tabOf(id: string): Workspace | undefined {
		return tabs.find((t) => t.id === id || !!findNode(t.root, id));
	}

	function freeName(): string {
		const taken = new Set(tabs.map((t) => t.name));
		for (let n = 1; ; n += 1) if (!taken.has(`Tab ${n}`)) return `Tab ${n}`;
	}

	/**
	 * Lift `id` out of the tree, closing up behind it: a split left with one child is replaced BY
	 * that child, because a split of one is not a split. Returns the subtree, or null when `id` is
	 * the last panel on the last tab — the one thing that has nowhere to go.
	 */
	function take(id: string): LayoutNode | null {
		const tab = tabOf(id);
		if (!tab) return null;
		if (tab.root.id === id) {
			if (tabs.length <= 1) return null;
			tabs = tabs.filter((t) => t !== tab);
			return tab.root;
		}
		const at = findParent(tab.root, id);
		if (!at) return null;
		const [gone] = at.parent.children.splice(at.index, 1);
		at.parent.sizes.splice(at.index, 1);
		normalize(at.parent.sizes);
		if (at.parent.children.length === 1) replace(tab, at.parent.id, at.parent.children[0]);
		return gone;
	}

	/** Put `node` beside `target`, splitting along `direction`; `share` is the newcomer's. */
	function insertBeside(
		node: LayoutNode,
		target: string,
		direction: Direction,
		before: boolean,
		share: number
	): boolean {
		const tab = tabOf(target);
		if (!tab) return false;
		const at = findParent(tab.root, target);
		const host = at?.parent;
		if (host && host.direction === direction) {
			const i = at.index + (before ? 0 : 1);
			host.children.splice(i, 0, node);
			host.sizes.splice(i, 0, share);
			normalize(host.sizes);
			return true;
		}
		const existing = findNode(tab.root, target);
		if (!existing) return false;
		const split: LayoutNode = {
			kind: 'split',
			id: mint('split'),
			direction,
			children: before ? [node, existing] : [existing, node],
			sizes: before ? [share, 1 - share] : [1 - share, share]
		};
		replace(tab, target, split);
		return true;
	}

	/** Swap the node with id `id` for `next`, wherever it sits — including a tab's root. */
	function replace(tab: Workspace, id: string, next: LayoutNode): void {
		if (tab.root.id === id) {
			tab.root = next;
			return;
		}
		const at = findParent(tab.root, id);
		if (at) at.parent.children[at.index] = next;
	}

	function normalize(sizes: number[]): void {
		const total = sizes.reduce((a, b) => a + b, 0);
		if (total > 0) for (let i = 0; i < sizes.length; i++) sizes[i] /= total;
	}

	return {
		// --- tabs ---------------------------------------------------------------------------------
		async addTab(o): Promise<TabRef | null> {
			const root = panel(o?.panelType ?? fresh);
			const tab: Workspace = { id: mint('tab'), name: freeName(), root };
			tabs.splice(o?.index ?? tabs.length, 0, tab);
			commit();
			return { tab: tab.id, panel: root.id };
		},

		async removeTab(tab) {
			if (tabs.length <= 1 || !tabs.some((t) => t.id === tab)) return false;
			tabs = tabs.filter((t) => t.id !== tab);
			return commit();
		},

		async renameTab(tab, name) {
			const t = tabs.find((x) => x.id === tab);
			const trimmed = name.trim();
			if (!t || !trimmed || tabs.some((x) => x !== t && x.name === trimmed)) return false;
			t.name = trimmed;
			return commit();
		},

		async reorderTab(tab, toIndex) {
			const from = tabs.findIndex((t) => t.id === tab);
			if (from < 0) return false;
			const [moved] = tabs.splice(from, 1);
			tabs.splice(Math.max(0, Math.min(toIndex, tabs.length)), 0, moved);
			return commit();
		},

		// --- panels -------------------------------------------------------------------------------
		async splitPanel(target, direction, placeBefore, ratio) {
			const born = panel(empty);
			if (!insertBeside(born, target, direction, placeBefore, ratio)) return null;
			commit();
			return born.id;
		},

		async removePanel(target) {
			return take(target) === null ? false : commit();
		},

		async resizeSplit(split, fractions) {
			const node = subtreeOf(split);
			if (node?.kind !== 'split' || node.children.length !== fractions.length) return false;
			node.sizes = [...fractions];
			normalize(node.sizes);
			return commit();
		},

		async setPanel(target, patch) {
			const node = subtreeOf(target);
			if (node?.kind !== 'panel') return false;
			// A new TYPE clears the old type's state: the two are one thing, and a stale bag under a
			// panel that no longer reads it is a value nothing can ever remove.
			if (patch.type !== undefined && patch.type !== node.panelType) {
				node.panelType = patch.type;
				node.state = undefined;
			}
			if (patch.state) node.state = { ...(node.state as object), ...patch.state };
			return commit();
		},

		async movePanel(subtree, to: Landing) {
			if ('panel' in to) {
				const source = subtreeOf(subtree);
				// Onto itself, or into its own descendant — the second would make a cycle.
				if (!source || subtree === to.panel || findNode(source, to.panel)) return false;
			}
			// Lifted FIRST, as any split-aware planner must: closing up behind the source can promote
			// a sibling into the very slot the newcomer is about to share.
			const moved = take(subtree);
			if (!moved) return false;
			if ('newTab' in to) {
				tabs.splice(to.newTab, 0, { id: mint('tab'), name: freeName(), root: moved });
				return commit();
			}
			if (!insertBeside(moved, to.panel, to.direction, to.placeBefore, 0.5)) return false;
			return commit();
		}
	};

	function subtreeOf(id: string): LayoutNode | null {
		const tab = tabOf(id);
		return (tab && findNode(tab.root, id)) ?? null;
	}

	/** Continue a handed-over tree's numbering rather than colliding with it. */
	function highestSeq(from: Workspace[]): number {
		let top = 0;
		for (const t of from)
			for (const id of [t.id, ...ids(t.root)]) {
				const n = Number(id.split('-').pop());
				if (Number.isFinite(n)) top = Math.max(top, n);
			}
		return top;
	}

	function ids(node: LayoutNode): string[] {
		return node.kind === 'panel' ? [node.id] : [node.id, ...node.children.flatMap(ids)];
	}
}
