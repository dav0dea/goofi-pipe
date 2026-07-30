/**
 * What the completion source needs to know about the patch — the whole of it.
 *
 * The expression scope is `{nd, t, np, globals}` (`crates/goofi-py/src/expr.rs`), so this is a small,
 * fully knowable surface rather than "general Python autocomplete": node names, each node's OUTPUT
 * slots, and the patch's global keys. Everything here already lives in the frontend, so a completion
 * is a synchronous local read and needs no RPC.
 *
 * It is an INTERFACE with a live implementation rather than a direct store read at the call site for
 * one reason: the completion source is the piece that must be unit-tested (spec §3), and a component
 * that mounts a graph store cannot be. The editor passes `liveCatalogue`; the tests pass a fake.
 */
import { graph } from '$lib/stores/graph.svelte';

export interface CatalogueSlot {
	name: string;
	/** The slot's declared dtype (`array` / `string` / `table`) — the completion's detail line. */
	dtype: string;
}

export interface CatalogueNode {
	/** The DISPLAY name, which is what `nd()` takes — flat and unique at every nesting depth. */
	name: string;
	/** Output slots, in declaration order. Its LENGTH is load-bearing: a single-output node may be
	 *  used bare, a multi-output one raises unless a slot is named (D-X10). */
	slots: CatalogueSlot[];
}

export interface CatalogueGlobal {
	name: string;
	/** `float` / `int` / `bool` / `string` — the completion's detail line. */
	type: string;
}

export interface ExprCatalogue {
	nodes: CatalogueNode[];
	globals: CatalogueGlobal[];
}

/** The live patch, read from the doc-authoritative graph store at the moment a completion is asked
 *  for — so a node added or renamed while the editor is open completes correctly with no wiring. */
export function liveCatalogue(): ExprCatalogue {
	const g = graph();
	return {
		nodes: g.nodes.map((n) => ({
			name: n.name,
			slots: Object.entries(n.output_slots).map(([name, dtype]) => ({ name, dtype }))
		})),
		globals: g.globals.map((gv) => ({ name: gv.name, type: gv.type }))
	};
}
