/**
 * The node a panel binds, expressed as a `Select`'s three inputs.
 *
 * Four panel types bind a node (`acceptsNode` in the manager's vocab: parameters, viewer, metadata,
 * console) and every one of them offers the SAME picker — so the list is built once here, pure and
 * unit-tested, and `NodeSelect.svelte` is only the store wiring around it.
 *
 * The committed value is the node's UID, never its display name: a uid is the panel binding's
 * identity (`panelState.linkedNodeName`) and survives a rename, while the label the user reads is
 * the display name the canvas paints on the node itself.
 */

/** One selectable node — everything the picker needs of it. */
export interface NodeChoice {
	uid: string;
	name: string;
}

export interface NodePickList {
	/** What the dropdown shows as selected. */
	value: string;
	/** Option keys, in list order. */
	options: string[];
	/** Display text per option key. */
	labels: Record<string, string>;
}

/** The "nothing bound" option. EMPTY on purpose: `Select` prepends a truthy value that is not among
 * its options, and an empty one is the single value it will not — so this option can never collide
 * with a live uid. */
export const NO_NODE = '';

/** Collate the way a user reads a name that ends in a number: `oscillator2` before `oscillator10`. */
const byName = (a: NodeChoice, b: NodeChoice): number =>
	a.name.localeCompare(b.name, undefined, { numeric: true });

export function nodePickList(
	nodes: readonly NodeChoice[],
	bound: string | null,
	emptyLabel: string
): NodePickList {
	const sorted = [...nodes].sort(byName);
	const labels: Record<string, string> = { [NO_NODE]: emptyLabel };
	for (const n of sorted) labels[n.uid] = n.name;
	return {
		// A binding whose node is gone reads as unbound. The engine clears the binding on removal,
		// but this replica can be a round trip behind it — and a `.gfi` can name a node the patch no
		// longer has — so the picker never has to draw a bare uid to say "that node is missing".
		value: bound && sorted.some((n) => n.uid === bound) ? bound : NO_NODE,
		options: [NO_NODE, ...sorted.map((n) => n.uid)],
		labels
	};
}
