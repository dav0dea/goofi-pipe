/**
 * The pure assembly that builds render-time `InstanceInfo` records for the sub-patch forest from the
 * CRDT doc (Phase-2 instances read cutover — see the plan doc). Kept Svelte-free and doc-handle-free
 * so it unit-tests directly; the store gathers `instanceViews(doc)` + node identities + a runtime
 * error lookup and calls this.
 *
 * Flat model: the doc mirror persists only a scope's STRUCTURAL fields (`name`, `parent`, `pos`,
 * `members: uid→{is_instance}`, `stubs: id→{dir,dtype,name,pos,inner_node?,inner_slot?}` — read into
 * the view's `interface`). No sharing (no def_id/kind/siblings). Every other render field is
 * reconstructed here to MATCH the backend's `describe_instance`:
 *
 *   slots     — a stub is a slot iff wired (`inner_node != null`), keyed `stub_id → dtype`, by `dir`.
 *   is_instance — a member uid that is itself a nested scope.
 *   error     — runtime (first errored descendant); supplied by the caller's overlay, NOT the doc.
 *   viewers   — a backend stub ({} end-to-end today).
 *
 * ROOT is NOT mirrored (the manager writes only real scope uids), so `assembleRoot` synthesizes it
 * exactly as the backend's `root_instance`: its members are every TOP-LEVEL entity keyed by uid.
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

/** Assemble ONE sub-patch scope from its doc view + the live scope-uid set (for `members.is_instance`)
 * + its runtime error overlay. Pure. Members are keyed by uid (flat model: no template locals). */
export function assembleInstance(
	view: InstanceView,
	instanceUids: Set<string>,
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
	for (const [uid, isInst] of Object.entries(view.members)) {
		members[uid] = { uid, is_instance: isInst || instanceUids.has(uid) };
	}
	return {
		uid: view.uid,
		name: view.name,
		parent: view.parent,
		interface: iface,
		pos: view.pos,
		members,
		slots: { input, output },
		error,
		viewers: {}
	};
}

/** Synthesize the ROOT scope (the top-level canvas). Not mirrored — its members are every TOP-LEVEL
 * entity (a node/scope owned by no scope), keyed by uid, mirroring the backend's `root_instance`. */
export function assembleRoot(
	nodes: { uid: string; name: string }[],
	instances: InstanceView[]
): InstanceInfo {
	const owned = new Set<string>();
	for (const inst of instances) for (const uid of Object.keys(inst.members)) owned.add(uid);
	const members: InstanceInfo['members'] = {};
	for (const n of nodes) if (!owned.has(n.uid)) members[n.uid] = { uid: n.uid, is_instance: false };
	for (const inst of instances) if (!owned.has(inst.uid)) members[inst.uid] = { uid: inst.uid, is_instance: true };
	return {
		uid: ROOT_ID,
		name: 'root',
		parent: null,
		interface: {},
		pos: [0, 0],
		members,
		slots: { input: {}, output: {} },
		error: null,
		viewers: {}
	};
}

/** A scope's deep error = the first errored descendant across its subtree (mirroring the backend
 * `instance_error`): a plain member's runtime NODE error, or a nested scope's own derived error.
 * `null` when the whole subtree is healthy. Derived — the bridge only emits `error` keyed by a real
 * node uid, never a scope uid. */
export function instanceError(
	view: InstanceView,
	byUid: Map<string, InstanceView>,
	nodeError: (uid: string) => string | null
): string | null {
	for (const memberUid of Object.keys(view.members)) {
		const nested = byUid.get(memberUid);
		const e = nested ? instanceError(nested, byUid, nodeError) : nodeError(memberUid);
		if (e) return e;
	}
	return null;
}

/** Build the full render instances map (INCLUDING synthetic ROOT) from the doc's scope forest +
 * node identities + a NODE-error lookup (each scope's deep error is derived from its members).
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
		out[view.uid] = assembleInstance(view, instanceUids, instanceError(view, byUid, nodeError));
	}
	return out;
}
