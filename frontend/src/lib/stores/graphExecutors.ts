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
		const name = await deps.control.call<string>('add_node', {
			type: a.payload.type,
			category: a.payload.category,
			pos: a.payload.pos,
			inst_id: a.payload.instId,
			name: a.payload.assignedName
		});
		a.payload.assignedName = name ?? a.payload.assignedName;
	},
	async inverse(action, deps) {
		const a = as<'add_node'>(action);
		if (a.payload.assignedName) await deps.control.call('remove_node', { name: a.payload.assignedName });
	}
};

const removeNode: Executor = {
	async forward(action, deps) {
		const a = as<'remove_node'>(action);
		await deps.control.call('remove_node', { name: a.payload.name });
	},
	async inverse(action, deps) {
		const a = as<'remove_node'>(action);
		const n = a.payload.node;
		// Re-create with the SAME display name (reused once freed) + its params.
		await deps.control.call('add_node', {
			type: n.type,
			category: n.category,
			pos: n.pos,
			name: a.payload.name,
			params: paramValues(n)
		});
		for (const link of a.payload.links) await deps.control.call('add_link', { ...link });
		// Panel-binding restoration (boundPanels) is wired in Phase 3.
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
		await deps.control.call('set_node_pos', { name: a.payload.name, pos: a.payload.newPos });
	},
	async inverse(action, deps) {
		const a = as<'set_node_pos'>(action);
		await deps.control.call('set_node_pos', { name: a.payload.name, pos: a.payload.oldPos });
	}
};

export const graphExecutors: Record<string, Executor> = {
	add_node: addNode,
	remove_node: removeNode,
	add_link: addLink,
	remove_link: removeLink,
	update_param: updateParam,
	set_expression: setExpression,
	set_node_pos: setNodePos
};
