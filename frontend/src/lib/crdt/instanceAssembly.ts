/**
 * The pure assembly that builds render-time `InstanceInfo` records for the sub-patch forest from the
 * CRDT doc (Phase-2 instances read cutover — see the plan doc). Kept Svelte-free and doc-handle-free
 * so it unit-tests directly; the store gathers `instanceViews(doc)` + node identities + a runtime
 * error lookup and calls this.
 *
 * The doc mirror persists only the STRUCTURAL fields of a real instance (`name`, `parent`, `def_id?`
 * when shared, `pos`, `members: local→uid`, `interface: bnd→{dir,dtype,name,pos,inner_node?,inner_slot?}`).
 * Every other render field is reconstructed here to MATCH the backend's `describe_instance`:
 *
 *   kind      — `def_id` present (⇔ shared, refcount>1) → 'shared', else 'unique'.
 *   slots     — a boundary is a slot iff WIRED (`inner_node != null`, i.e. `resolve_boundary` hit),
 *               keyed `bnd_id → dtype`, split by `dir`.
 *   siblings  — other instances sharing the same `def_id` (shared family); [] when unique.
 *   is_instance — a member uid that is itself a live instance.
 *   error     — runtime (first errored descendant); supplied by the caller's overlay, NOT the doc.
 *   viewers   — a backend stub ({} end-to-end today); matched as {} until real per-boundary
 *               persistence lands (a pre-existing gap, out of scope for the read cutover).
 *
 * ROOT is NOT mirrored (the manager writes only real `instance_uids`), so `assembleRoot` synthesizes
 * it exactly as the backend's `root_instance`: its members are every TOP-LEVEL entity keyed by name.
 */
import type { InstanceInfo, SubPatchPort } from '$lib/api/control';
import type { InstanceView, BoundaryView } from './graphDoc';
import { ROOT_ID } from '$lib/editor/subpatchScene';

/** Map a doc `BoundaryView` to the render `SubPatchPort` (the interface value type). */
function toPort(b: BoundaryView): SubPatchPort {
	return {
		dir: b.dir === 'in' ? 'in' : 'out',
		dtype: b.dtype,
		inner_node: b.inner_node ?? null,
		inner_slot: b.inner_slot ?? null,
		pos: b.pos,
		name: b.name
	};
}

/** Assemble ONE real sub-patch instance from its doc view + the live instance set (for
 * `members.is_instance`) + all instance views (for `siblings`) + its runtime error overlay. Pure. */
export function assembleInstance(
	view: InstanceView,
	instanceUids: Set<string>,
	allInstances: InstanceView[],
	error: string | null
): InstanceInfo {
	const iface: Record<string, SubPatchPort> = {};
	const input: Record<string, string> = {};
	const output: Record<string, string> = {};
	for (const b of view.interface) {
		iface[b.bnd_id] = toPort(b);
		if (b.inner_node != null) {
			if (b.dir === 'in') input[b.bnd_id] = b.dtype;
			else output[b.bnd_id] = b.dtype;
		}
	}
	const members: InstanceInfo['members'] = {};
	for (const [local, uid] of Object.entries(view.members)) {
		members[local] = { uid, is_instance: instanceUids.has(uid) };
	}
	// `def_id` is doc-written only when shared, so its presence IS the shared flag; siblings are the
	// other holders of that def. A unique instance has no def_id and thus no siblings.
	const siblings =
		view.def_id === undefined
			? []
			: allInstances.filter((o) => o.uid !== view.uid && o.def_id === view.def_id).map((o) => o.uid);
	return {
		uid: view.uid,
		name: view.name,
		kind: view.def_id !== undefined ? 'shared' : 'unique',
		def_id: view.def_id ?? null,
		parent: view.parent,
		interface: iface,
		pos: view.pos,
		members,
		slots: { input, output },
		siblings,
		error,
		viewers: {}
	};
}

/** Synthesize the ROOT scope instance (the top-level canvas). Not mirrored — its members are every
 * TOP-LEVEL entity (a node/instance owned by no real instance), keyed by display name, mirroring the
 * backend's `root_instance`. Pure. */
export function assembleRoot(
	nodes: { uid: string; name: string }[],
	instances: InstanceView[]
): InstanceInfo {
	const owned = new Set<string>();
	for (const inst of instances) for (const uid of Object.values(inst.members)) owned.add(uid);
	const members: InstanceInfo['members'] = {};
	for (const n of nodes) if (!owned.has(n.uid)) members[n.name] = { uid: n.uid, is_instance: false };
	for (const inst of instances) if (!owned.has(inst.uid)) members[inst.name] = { uid: inst.uid, is_instance: true };
	return {
		uid: ROOT_ID,
		name: 'root',
		kind: 'unique',
		def_id: null,
		parent: null,
		interface: {},
		pos: [0, 0],
		members,
		slots: { input: {}, output: {} },
		siblings: [],
		error: null,
		viewers: {}
	};
}

/** An instance's deep error = the first errored descendant across its subtree (recursion-correct,
 * mirroring the backend `instance_error`): a plain member's runtime NODE error, or a nested
 * instance's own derived error. `null` when the whole subtree is healthy. Derived — NOT event-
 * overlaid: the bridge only ever emits `error` keyed by a real node uid, never an instance uid. */
function instanceError(
	view: InstanceView,
	byUid: Map<string, InstanceView>,
	nodeError: (uid: string) => string | null
): string | null {
	for (const memberUid of Object.values(view.members)) {
		const nested = byUid.get(memberUid);
		const e = nested ? instanceError(nested, byUid, nodeError) : nodeError(memberUid);
		if (e) return e;
	}
	return null;
}

/** Build the full render instances map (INCLUDING synthetic ROOT) from the doc's instance forest +
 * node identities + a NODE-error lookup (each instance's deep error is derived from its members).
 * Pure — the store gathers the inputs and calls this. */
export function assembleInstances(
	instances: InstanceView[],
	nodes: { uid: string; name: string }[],
	nodeError: (uid: string) => string | null
): Record<string, InstanceInfo> {
	const instanceUids = new Set(instances.map((i) => i.uid));
	const byUid = new Map(instances.map((i) => [i.uid, i]));
	const out: Record<string, InstanceInfo> = { [ROOT_ID]: assembleRoot(nodes, instances) };
	for (const view of instances) {
		out[view.uid] = assembleInstance(view, instanceUids, instances, instanceError(view, byUid, nodeError));
	}
	return out;
}
