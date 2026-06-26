/**
 * Graph-domain executors. Each forward/inverse issues plain RPCs through the
 * injected control client; the server-authoritative store reconciles via the
 * echo events any user action would receive. Pre-state lives in the action
 * payload (captured by the recording wrappers in `graph.svelte.ts`), so these
 * functions are pure of the store except where they must read live params.
 *
 * Per-kind recipes are specified in the design spec §3.
 */
import { paramValues } from '$lib/api/control';
import type { Executor, ExecutorDeps, ExprState, GraphAction } from './history.svelte';

/** Narrow a generic Action to a specific graph kind for an executor body. */
function as<K extends GraphAction['kind']>(a: unknown): Extract<GraphAction, { kind: K }> {
	return a as Extract<GraphAction, { kind: K }>;
}

const addNode: Executor = {
	async forward(action, deps: ExecutorDeps) {
		const a = as<'add_node'>(action);
		// On redo, restore the SAME uid (member_uid) + display name so links/panels
		// that reference this node reconnect to it.
		const uid = await deps.control.call<string>('add_node', {
			type: a.payload.type,
			category: a.payload.category,
			pos: a.payload.pos,
			inst_id: a.payload.instId,
			name: a.payload.name,
			member_uid: a.payload.uid
		});
		a.payload.uid = uid ?? a.payload.uid;
	},
	async inverse(action, deps) {
		const a = as<'add_node'>(action);
		if (a.payload.uid) await deps.control.call('remove_node', { node: a.payload.uid });
	}
};

const removeNode: Executor = {
	async forward(action, deps) {
		const a = as<'remove_node'>(action);
		await deps.control.call('remove_node', { node: a.payload.uid });
	},
	async inverse(action, deps) {
		const a = as<'remove_node'>(action);
		const n = a.payload.node;
		// Re-create with the SAME uid (so uid-keyed links/panels reconnect) + display
		// name + params.
		await deps.control.call('add_node', {
			type: n.type,
			category: n.category,
			pos: n.pos,
			name: n.name,
			member_uid: a.payload.uid,
			params: paramValues(n)
		});
		for (const link of a.payload.links) await deps.control.call('add_link', { ...link });
		// Re-bind any panels that were emptied when the node was deleted.
		for (const bp of a.payload.boundPanels) deps.workspace.setPanelState(bp.panelId, bp.state);
	}
};

const renameNode: Executor = {
	async forward(action, deps) {
		const a = as<'rename_node'>(action);
		await deps.control.call('rename_node', { node: a.payload.uid, name: a.payload.newName });
	},
	async inverse(action, deps) {
		const a = as<'rename_node'>(action);
		await deps.control.call('rename_node', { node: a.payload.uid, name: a.payload.oldName });
	}
};

const addLink: Executor = {
	async forward(action, deps) {
		const a = as<'add_link'>(action);
		await deps.control.call('add_link', { ...a.payload.link });
	},
	async inverse(action, deps) {
		const a = as<'add_link'>(action);
		await deps.control.call('remove_link', { ...a.payload.link });
		if (a.payload.displaced) await deps.control.call('add_link', { ...a.payload.displaced });
	}
};

const removeLink: Executor = {
	async forward(action, deps) {
		const a = as<'remove_link'>(action);
		await deps.control.call('remove_link', { ...a.payload.link });
	},
	async inverse(action, deps) {
		const a = as<'remove_link'>(action);
		await deps.control.call('add_link', { ...a.payload.link });
	}
};

function paramCall(node: string, group: string, name: string, value: unknown): Record<string, unknown> {
	return { node, group, name, value };
}

const updateParam: Executor = {
	async forward(action, deps) {
		const a = as<'update_param'>(action);
		await deps.control.call('update_param', paramCall(a.payload.node, a.payload.group, a.payload.name, a.payload.newValue));
	},
	async inverse(action, deps) {
		const a = as<'update_param'>(action);
		await deps.control.call('update_param', paramCall(a.payload.node, a.payload.group, a.payload.name, a.payload.oldValue));
	}
};

function exprCall(node: string, group: string, name: string, e: ExprState): Record<string, unknown> {
	return {
		node,
		group,
		name,
		expression: e.expression,
		expression_enabled: e.enabled,
		expression_triggers_process: e.triggers_process,
		expression_autoeval: e.autoeval
	};
}

const setExpression: Executor = {
	async forward(action, deps) {
		const a = as<'set_expression'>(action);
		await deps.control.call('set_expression', exprCall(a.payload.node, a.payload.group, a.payload.name, a.payload.newExpr));
	},
	async inverse(action, deps) {
		const a = as<'set_expression'>(action);
		await deps.control.call('set_expression', exprCall(a.payload.node, a.payload.group, a.payload.name, a.payload.oldExpr));
	}
};

const setNodePos: Executor = {
	async forward(action, deps) {
		const a = as<'set_node_pos'>(action);
		await deps.control.call('set_node_pos', { node: a.payload.uid, pos: a.payload.newPos });
	},
	async inverse(action, deps) {
		const a = as<'set_node_pos'>(action);
		await deps.control.call('set_node_pos', { node: a.payload.uid, pos: a.payload.oldPos });
	}
};

// --- composite / sub-patch -------------------------------------------------

const groupNodes: Executor = {
	async forward(action, deps) {
		const a = as<'group_nodes'>(action);
		const r = await deps.control.call<{ inst_id: string }>('group_nodes', {
			members: a.payload.members,
			pos: a.payload.pos
		});
		if (r?.inst_id) a.payload.instId = r.inst_id; // remap for the next inverse
	},
	async inverse(action, deps) {
		const a = as<'group_nodes'>(action);
		await deps.control.call('expand_instance', { inst_id: a.payload.instId });
	}
};

const expandInstance: Executor = {
	async forward(action, deps) {
		const a = as<'expand_instance'>(action);
		await deps.control.call('expand_instance', { inst_id: a.payload.instId });
	},
	async inverse(action, deps) {
		const a = as<'expand_instance'>(action);
		const r = await deps.control.call<{ inst_id: string }>('group_nodes', {
			members: a.payload.restoredMembers
		});
		if (r?.inst_id) a.payload.instId = r.inst_id; // the re-grouped id (for redo)
	}
};

const addBoundary: Executor = {
	async forward(action, deps) {
		const a = as<'add_boundary'>(action);
		const r = await deps.control.call<{ bnd_id: string }>('add_boundary', {
			inst_id: a.payload.instId,
			dir: a.payload.dir,
			dtype: a.payload.dtype,
			pos: a.payload.pos
		});
		if (r?.bnd_id) a.payload.bndId = r.bnd_id;
	},
	async inverse(action, deps) {
		const a = as<'add_boundary'>(action);
		await deps.control.call('remove_boundary', { inst_id: a.payload.instId, bnd_id: a.payload.bndId });
	}
};

const wireBoundary: Executor = {
	async forward(action, deps) {
		const a = as<'wire_boundary'>(action);
		await deps.control.call('wire_boundary', {
			inst_id: a.payload.instId,
			bnd_id: a.payload.bndId,
			inner_node: a.payload.newInner.node,
			inner_slot: a.payload.newInner.slot
		});
	},
	async inverse(action, deps) {
		const a = as<'wire_boundary'>(action);
		await deps.control.call('wire_boundary', {
			inst_id: a.payload.instId,
			bnd_id: a.payload.bndId,
			inner_node: a.payload.oldInner.node,
			inner_slot: a.payload.oldInner.slot
		});
	}
};

const removeBoundary: Executor = {
	async forward(action, deps) {
		const a = as<'remove_boundary'>(action);
		await deps.control.call('remove_boundary', { inst_id: a.payload.instId, bnd_id: a.payload.bndId });
	},
	async inverse(action, deps) {
		const a = as<'remove_boundary'>(action);
		const p = a.payload.port;
		const r = await deps.control.call<{ bnd_id: string }>('add_boundary', {
			inst_id: a.payload.instId,
			dir: p.dir,
			dtype: p.dtype,
			pos: p.pos
		});
		const bndId = r?.bnd_id ?? a.payload.bndId;
		if (p.inner_node) {
			await deps.control.call('wire_boundary', {
				inst_id: a.payload.instId,
				bnd_id: bndId,
				inner_node: p.inner_node,
				inner_slot: p.inner_slot
			});
		}
	}
};

const setBoundaryPos: Executor = {
	async forward(action, deps) {
		const a = as<'set_boundary_pos'>(action);
		await deps.control.call('set_boundary_pos', { inst_id: a.payload.instId, bnd_id: a.payload.bndId, pos: a.payload.newPos });
	},
	async inverse(action, deps) {
		const a = as<'set_boundary_pos'>(action);
		await deps.control.call('set_boundary_pos', { inst_id: a.payload.instId, bnd_id: a.payload.bndId, pos: a.payload.oldPos });
	}
};

const duplicateShared: Executor = {
	async forward(action, deps) {
		const a = as<'duplicate_shared'>(action);
		const r = await deps.control.call<{ inst_id: string }>('duplicate_shared', {
			inst_id: a.payload.instId,
			pos: a.payload.pos
		});
		if (r?.inst_id) a.payload.newInstId = r.inst_id;
	},
	async inverse(action, deps) {
		const a = as<'duplicate_shared'>(action);
		// remove_node routes to remove_instance for an instance id (bridge).
		if (a.payload.newInstId) await deps.control.call('remove_node', { node: a.payload.newInstId });
		// If the source was unique before, sharing it left a residual definition —
		// detach it again so the graph matches the pre-duplicate state.
		if (a.payload.wasUnique) await deps.control.call('make_unique', { inst_id: a.payload.instId });
	}
};

const makeUnique: Executor = {
	async forward(action, deps) {
		const a = as<'make_unique'>(action);
		await deps.control.call('make_unique', { inst_id: a.payload.instId });
	},
	async inverse(action, deps) {
		const a = as<'make_unique'>(action);
		// KNOWN LIMITATION (backlog #21, spec §11 deferral): undoing make_unique
		// re-promotes the instance to shared, but `duplicate_shared` mints a FRESH
		// definition rather than re-attaching to the exact prior `defIdBefore` and
		// its original siblings. So the instance is shared again (not left unique),
		// but its shared-group identity differs from before the make_unique. An
		// exact re-attach needs a backend "re-share to existing def_id" op, which is
		// deliberately deferred — restoring a node to a specific prior definition
		// identity is a larger sub-patch-runtime change than this frontend layer owns.
		if (a.payload.defIdBefore) await deps.control.call('duplicate_shared', { inst_id: a.payload.instId });
	}
};

const loadPatch: Executor = {
	async forward(action, deps) {
		const a = as<'load_patch'>(action);
		await deps.graph.applyLoadCheckpoint(a.payload.afterYaml);
		if (a.payload.afterLayout) deps.workspace.restore(a.payload.afterLayout);
	},
	async inverse(action, deps) {
		const a = as<'load_patch'>(action);
		await deps.graph.applyLoadCheckpoint(a.payload.beforeYaml);
		// Force the exact prior layout (the YAML reload re-hydrates a layout too,
		// but the snapshot guarantees the precise prior arrangement + nav).
		if (a.payload.beforeLayout) deps.workspace.restore(a.payload.beforeLayout);
	}
};

export const graphExecutors: Record<string, Executor> = {
	add_node: addNode,
	load_patch: loadPatch,
	remove_node: removeNode,
	add_link: addLink,
	remove_link: removeLink,
	update_param: updateParam,
	set_expression: setExpression,
	set_node_pos: setNodePos,
	rename_node: renameNode,
	group_nodes: groupNodes,
	expand_instance: expandInstance,
	add_boundary: addBoundary,
	wire_boundary: wireBoundary,
	remove_boundary: removeBoundary,
	set_boundary_pos: setBoundaryPos,
	duplicate_shared: duplicateShared,
	make_unique: makeUnique
};
