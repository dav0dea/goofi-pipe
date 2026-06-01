/**
 * Graph selection — hoisted out of the (former) Editor monolith so it can be
 * shared across panels. The node editor writes it; the Parameters / Metadata
 * panels and the auto side-panel read `selectedNode` to follow the focus.
 *
 * Sets are replaced (never mutated in place) so a plain assignment drives
 * reactivity, matching how the old Editor managed `selection`/`edgeSelection`.
 * The click semantics below are preserved verbatim from that code, including
 * the asymmetry that a plain node click leaves edge selection untouched while
 * a plain edge click clears node selection.
 */
import { graph } from './graph.svelte';
import type { NodeInstanceInfo } from '$lib/api/control';

class SelectionStore {
	nodes = $state<Set<string>>(new Set());
	edges = $state<Set<string>>(new Set());

	/** The single selected node, or null when zero / many are selected. */
	get selectedNode(): NodeInstanceInfo | null {
		if (this.nodes.size !== 1) return null;
		const name = [...this.nodes][0];
		return graph().nodes.find((n) => n.name === name) ?? null;
	}

	// --- node selection ----------------------------------------------------

	clickNode(name: string, additive: boolean): void {
		if (additive) {
			const next = new Set(this.nodes);
			if (next.has(name)) next.delete(name);
			else next.add(name);
			this.nodes = next;
		} else {
			this.nodes = new Set([name]);
		}
	}

	selectNodes(names: Iterable<string>): void {
		this.nodes = new Set(names);
	}

	// --- edge selection ----------------------------------------------------

	clickEdge(id: string, additive: boolean): void {
		if (additive) {
			const next = new Set(this.edges);
			if (next.has(id)) next.delete(id);
			else next.add(id);
			this.edges = next;
		} else {
			this.edges = new Set([id]);
			this.nodes = new Set();
		}
	}

	// --- clearing ----------------------------------------------------------

	/** Empty-canvas click: clear everything unless the user is shift-extending. */
	clickPane(shift: boolean): void {
		if (shift) return;
		this.clear();
	}

	clearNodes(): void {
		if (this.nodes.size) this.nodes = new Set();
	}
	clearEdges(): void {
		if (this.edges.size) this.edges = new Set();
	}
	clear(): void {
		this.clearNodes();
		this.clearEdges();
	}
}

let _sel: SelectionStore | null = null;
export function selection(): SelectionStore {
	if (!_sel) _sel = new SelectionStore();
	return _sel;
}
