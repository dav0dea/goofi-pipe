/** What the expression completion source knows about the patch: node names, output slots, globals. */
import { graph } from '$lib/stores/graph.svelte';

export interface CatalogueSlot {
	name: string;
	dtype: string;
}

export interface CatalogueNode {
	/** The DISPLAY name, which is what `nd()` takes. */
	name: string;
	/** Output slots; the LENGTH is load-bearing — a multi-output node raises unless a slot is named. */
	slots: CatalogueSlot[];
}

export interface CatalogueGlobal {
	name: string;
	type: string;
}

export interface ExprCatalogue {
	nodes: CatalogueNode[];
	globals: CatalogueGlobal[];
}

/** The live patch, read at the moment a completion is asked for. */
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
