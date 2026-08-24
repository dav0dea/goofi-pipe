/** What the expression completion source knows about the patch: node names, output slots, globals. */
import { graph, type GraphStore } from '$lib/stores/graph.svelte';

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

/** The live patch, read at the moment a completion is asked for. `g` is the store to read, so a
 * test can drive this against a seeded one. */
export function liveCatalogue(g: GraphStore = graph()): ExprCatalogue {
	return {
		// Everything `nd()` can name — leaves, boundary ports and sub-patch facades alike. A facade
		// keys its slots by port uid, and `nd()` addresses them by the port's NAME, so the label is
		// what is offered.
		nodes: g.bindable.flatMap(({ uid }) => {
			const n = g.nodeById(uid);
			if (!n) return [];
			return [{
				name: n.name,
				slots: Object.entries(n.output_slots).map(([key, dtype]) => ({
					name: n.slot_labels?.[key] ?? key,
					dtype
				}))
			}];
		}),
		globals: g.globals.map((gv) => ({ name: gv.name, type: gv.type }))
	};
}
