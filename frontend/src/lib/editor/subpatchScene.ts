/**
 * Pure scene algebra for rendering a recursive sub-patch tree on the canvas.
 *
 * The backend ships a recursive instance tree (each instance has a `parent` and
 * `members` keyed by template-local). The editor renders ONE scope at a time — the
 * direct children of the entered instance — drawing real nodes and nested instances as
 * peers, and rerouting any link that crosses a sub-patch boundary to the nearest VISIBLE
 * boundary port. The root patch is just another scope: ROOT_ID names the first-class
 * root instance (parent=null, no boundaries) whose members are every top-level entity, so
 * there is no null special case — `scope` is always a real instance id. This module is the
 * testable core of that mapping; `NodeEditorPanel.svelte` is a thin descriptor->SvelteFlow
 * adapter over it.
 *
 * Kept free of Svelte/rune types so it unit-tests with plain objects (CLAUDE.md: rune
 * glue can't mount in vitest — keep the logic in a .ts module that is unit-tested).
 */

/**
 * The reserved id of the root scope. MUST match `goofi.manager.ROOT_ID` — the backend
 * ships ROOT as an instance and reports every top-level entity's membership/parent against
 * it, so the editor's root scope is literally this id (never `null`). Lives here, in the
 * pure scope-algebra module, so it stays dependency-light and co-located with its tests.
 */
export const ROOT_ID = '__root__';

/**
 * Boundary-pill node id codec — the SINGLE source of truth for how a sub-patch's
 * In/Out boundary is given a SvelteFlow node/handle id. A boundary isn't a real node,
 * so it has no uid; inside an entered scope it's drawn as a pill whose id encodes the
 * owning instance + the interface (boundary) key, so the panel can map a drag/connect/
 * delete on that pill back to (instId, bnd). Kept here (pure, tested) rather than as
 * magic strings in the component.
 */
const BND_PREFIX = '__bnd__';
const BND_SEP = '::';

/** The flow-node id for instance `instId`'s boundary `bnd`. */
export function boundaryNodeId(instId: string, bnd: string): string {
	return `${BND_PREFIX}${instId}${BND_SEP}${bnd}`;
}

/** Whether `id` is ANY boundary-pill id (scope-agnostic) — e.g. to exclude pills from
 * a select-all of real members, or detect a boundary in a drag set. */
export function isBoundaryNodeId(id: string): boolean {
	return id.startsWith(BND_PREFIX);
}

/** The boundary (interface) key of a pill id that belongs to the ENTERED scope, or null
 * — a non-boundary id, a pill from another instance, or no scope entered. Scope-checked
 * (matches the full `__bnd__<entered>::` prefix) so a cross-instance id can never be
 * mis-sliced into a garbage key. */
export function parseBoundaryNodeId(id: string, entered: string | null): string | null {
	if (!entered) return null;
	const prefix = `${BND_PREFIX}${entered}${BND_SEP}`;
	return id.startsWith(prefix) ? id.slice(prefix.length) : null;
}

export interface ScenePort {
	dir: 'in' | 'out';
	inner_node: string | null; // a member's template-local (node OR nested instance), or null when unwired
	inner_slot: string | null; // the member's slot, or — for a nested-instance member — that instance's boundary id
}

export interface SceneInstance {
	parent: string | null;
	/** template-local -> { member uid, whether the member is itself a nested instance } */
	members: Record<string, { uid: string; is_instance: boolean }>;
	interface: Record<string, ScenePort>;
}

/** What an entity (node or nested instance) is, indexed by uid. */
export interface MemberEntry {
	instId: string; // the instance that directly contains this entity
	local: string; // its template-local within that instance
	// Carried for parity with the snapshot's member shape; node-vs-instance classification
	// for rendering is done by childrenOfScope (real nodes vs Object.keys(instances)).
	is_instance: boolean;
}

/** Index every entity (node OR nested instance) by uid -> {instId, local, is_instance}.
 * Every non-root entity appears in exactly one instance's `members`, so this is the
 * single source for both "who is my parent scope" and "what is my local". */
export function buildMemberIndex(instances: Record<string, SceneInstance>): Map<string, MemberEntry> {
	const idx = new Map<string, MemberEntry>();
	for (const [instId, inst] of Object.entries(instances)) {
		for (const [local, m] of Object.entries(inst.members)) {
			idx.set(m.uid, { instId, local, is_instance: m.is_instance });
		}
	}
	return idx;
}

/** The parent-scope uid of any entity, or null at the top level. A node's parent is
 * the instance whose members include it; an instance's parent is its `.parent` — both
 * are captured by the member index (a nested instance is a member too). */
export function parentOf(uid: string, index: Map<string, MemberEntry>): string | null {
	return index.get(uid)?.instId ?? null;
}

/** The direct children of a scope: real-node uids and nested-instance uids whose
 * parent === scope (`scope` is a real instance id — ROOT_ID for the root patch). This is
 * what the canvas renders at the entered depth — a nested instance shows up ONLY inside
 * its parent, never leaked to the root; ROOT (parent=null) is never its own child. */
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

/** Resolve a link endpoint (a real node + slot) to what's actually DRAWN in `scope`:
 * walk up the nesting tree from the endpoint to the nearest entity whose parent is the
 * entered scope, rerouting through each level's boundary port. Returns:
 *  - {node, handle} when the endpoint (or an ancestor of it) is a direct child of scope;
 *  - null when a level doesn't expose the slot up the chain (purely-internal link, hidden);
 *  - null when the endpoint lives OUTSIDE the entered scope's subtree.
 * At the root scope (scope === ROOT_ID) this reduces to the identity for a top-level node
 * and to a one-hop climb for a member of a top-level instance — bit-identical to the
 * single-level case. */
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
		if (p === null) return null; // climbed to the root without hitting scope -> outside the subtree
		const local = index.get(curUid)?.local;
		const inst = instances[p];
		const bnd = Object.entries(inst.interface).find(
			([, port]) => port.dir === dir && port.inner_node === local && port.inner_slot === handle
		);
		if (!bnd) return null; // not exposed up the chain -> hidden
		handle = bnd[0];
		curUid = p;
	}
	return { node: curUid, handle };
}
