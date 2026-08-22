/** Pure scene algebra for rendering one scope of a recursive sub-patch tree on the canvas. */

/** The id the render tree gives the root scope. The document does not name it — a record simply
 * has no `scope` at the top level — so this is the frontend's own sentinel, not a wire contract. */
export const ROOT_ID = '__root__';

export interface ScenePort {
	dir: 'in' | 'out';
	inner_node: string | null; // the DIRECT inner member uid, or null when unwired
	inner_slot: string | null; // the member's slot, or a nested scope's stub id
}

export interface SceneInstance {
	parent: string | null;
	members: Record<string, { uid: string; is_instance: boolean }>;
	interface: Record<string, ScenePort>;
}

export interface MemberEntry {
	instId: string; // the scope that directly contains this entity
	is_instance: boolean;
}

/** Index every entity (node OR nested instance) by uid — the one source for its parent scope. */
export function buildMemberIndex(instances: Record<string, SceneInstance>): Map<string, MemberEntry> {
	const idx = new Map<string, MemberEntry>();
	for (const [instId, inst] of Object.entries(instances)) {
		for (const m of Object.values(inst.members)) {
			idx.set(m.uid, { instId, is_instance: m.is_instance });
		}
	}
	return idx;
}

/** The parent-scope uid of any entity, or null at the top level. */
export function parentOf(uid: string, index: Map<string, MemberEntry>): string | null {
	return index.get(uid)?.instId ?? null;
}

/** The direct children of a scope — what the canvas renders at the entered depth. */
export function childrenOfScope(
	scope: string,
	instances: Record<string, SceneInstance>,
	nodeUids: string[],
	index: Map<string, MemberEntry>
): { nodeUids: string[]; instUids: string[] } {
	return {
		nodeUids: nodeUids.filter((uid) => parentOf(uid, index) === scope),
		instUids: Object.keys(instances).filter((iid) => instances[iid].parent === scope)
	};
}

/** Resolve a link endpoint to what is actually DRAWN in `scope`, rerouting through each level's
 * boundary port; null when the slot is not exposed up the chain, or lies outside the subtree. */
export function drawEndpoint(
	uid: string,
	slot: string,
	dir: 'in' | 'out',
	scope: string,
	instances: Record<string, SceneInstance>,
	index: Map<string, MemberEntry>
): { node: string; handle: string } | null {
	let curUid = uid;
	let handle = slot;
	const seen = new Set<string>();
	while (parentOf(curUid, index) !== scope) {
		if (seen.has(curUid)) return null; // defensive against a corrupted cycle
		seen.add(curUid);
		const p = parentOf(curUid, index);
		if (p === null) return null; // outside the entered subtree
		const inst = instances[p];
		const bnd = Object.entries(inst.interface).find(
			([, port]) => port.dir === dir && port.inner_node === curUid && port.inner_slot === handle
		);
		if (!bnd) return null; // not exposed up the chain
		handle = bnd[0];
		curUid = p;
	}
	return { node: curUid, handle };
}
