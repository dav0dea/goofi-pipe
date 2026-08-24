/** Pure scene algebra for rendering one scope of a recursive sub-patch tree on the canvas. */

/** The id the render tree gives the root scope. The document does not name it — a record simply
 * has no `scope` at the top level — so this is the frontend's own sentinel, not a wire contract. */
export const ROOT_ID = '__root__';

/** The direct children of a scope — what the canvas renders at the entered depth. */
export function childrenOfScope(scope: string, index: Map<string, string>): string[] {
	return [...index].filter(([, parent]) => parent === scope).map(([uid]) => uid);
}

/** Resolve a link endpoint to what is actually DRAWN in `scope`: climb the scope chain, and each
 * level up the entity becomes its parent facade with the entity ITSELF as the handle, because a
 * port IS the facade's slot. Null when the endpoint lies outside the entered subtree. */
export function drawEndpoint(
	uid: string,
	slot: string,
	scope: string,
	index: Map<string, string>
): { node: string; handle: string } | null {
	let node = uid;
	let handle = slot;
	const seen = new Set<string>();
	while (index.get(node) !== scope) {
		const parent = index.get(node);
		if (parent === undefined || parent === ROOT_ID || seen.has(node)) return null;
		seen.add(node);
		handle = node;
		node = parent;
	}
	return { node, handle };
}
