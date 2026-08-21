/** Render-time `InstanceInfo` records for the sub-patch forest; what the doc does not mirror is
 * reconstructed here to MATCH the backend's `describe_instance`. */
import type { InstanceInfo, SubPatchPort } from '$lib/api/control';
import type { InstanceView, BoundaryView } from './graphDoc';
import { ROOT_ID } from '$lib/editor/subpatchScene';

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

/** Assemble ONE sub-patch scope from its doc view, the live scope-uid set and its runtime error overlay. */
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

/** Synthesize the ROOT scope: it is not mirrored, so its members are every entity no scope owns. */
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

/** A scope's deep error: the first errored descendant — derived, as the bridge keys `error` by node uid. */
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

/** Build the full render instances map, including the synthetic ROOT. */
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
