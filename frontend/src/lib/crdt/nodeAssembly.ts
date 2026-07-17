/**
 * The pure three-way merge that assembles a `NodeInstanceInfo` for rendering from its three
 * sources of truth (CRDT Phase-2 read cutover — see the plan doc). Kept Svelte-free and free of any
 * store/doc handle so it unit-tests directly; the reactive store gathers the three inputs and calls
 * this per node.
 *
 *   DOC     — existence, uid/type/name/pos, per-param value + expression binding (the merge-safe
 *             leaves clients write straight to the doc).
 *   CATALOG — the STATIC per-type shape from `list_nodes` (`NodeTypeInfo`): category, doc, slots,
 *             and each param's descriptor structure (type/vmin/vmax/trigger/doc/refreshable + the
 *             default options/value). The adopted "static PARAMS" node-authoring pattern guarantees
 *             a node instance's param group/name set equals its type's — so the catalog is a
 *             reliable structural source.
 *   RUNTIME — event-sourced state that never enters the doc: error/stage/crashed/stats/log_endpoint/
 *             membership, and per-param expression_error + refreshed StringParam options.
 */
import type { NodeInstanceInfo, NodeTypeInfo, NodeStage, NodeStats } from '$lib/api/control';
import type { ParamDescriptor } from '$lib/api/types';
import type { NodeView } from './graphDoc';

/** The doc-owned param leaves for one node: value + optional expression binding, per group/name. */
export type DocParamLeaves = Record<
	string,
	Record<
		string,
		{
			value?: number | string | boolean;
			expr?: { source: string; enabled: boolean; triggers: boolean };
		}
	>
>;

/** The persisted per-slot viewer blob (seeded into ui/inlineView by the store, not rendered here). */
export type ViewersBlob = NodeInstanceInfo['viewers'];

/** Per-param runtime overlay (event-sourced, never in the doc). */
export interface ParamRuntime {
	expression_error?: string | null;
	/** Refreshed StringParam options (device/stream pickers) — override the catalog default. */
	options?: string[] | null;
}

/** Node-level runtime overlay (event-sourced from state_update/error/node_stage/node_stats/…). */
export interface RuntimeOverlay {
	error?: string | null;
	stage?: NodeStage;
	crashed?: boolean;
	restarts?: number;
	crashExit?: number | null;
	stats?: NodeStats | null;
	log_endpoint?: string | null;
	membership?: { instance: string; local_name: string } | null;
	params?: Record<string, Record<string, ParamRuntime>>;
}

/** A descriptor for a param whose type is unknown (its node type is absent from the catalog — e.g.
 * missing deps). Renders the doc's value/binding with no bounds/options. */
function unknownParam(): ParamDescriptor {
	return {
		type: 'unknown',
		value: undefined,
		doc: null,
		save_param: true,
		refreshable: false,
		expression: null,
		expression_enabled: false,
		expression_triggers_process: false,
		expression_error: null
	};
}

/** Merge one param: catalog structure (or an unknown descriptor) + doc value/binding + runtime. */
function mergeParam(
	catalog: ParamDescriptor | undefined,
	leaf: DocParamLeaves[string][string] | undefined,
	runtime: ParamRuntime | undefined
): ParamDescriptor {
	// Shallow-copy so the shared catalog descriptor is never mutated.
	const p: ParamDescriptor = catalog ? { ...catalog } : unknownParam();
	// Doc leaves override the committed value + expression binding.
	if (leaf && leaf.value !== undefined) (p as { value: unknown }).value = leaf.value;
	if (leaf?.expr) {
		p.expression = leaf.expr.source;
		p.expression_enabled = leaf.expr.enabled;
		p.expression_triggers_process = leaf.expr.triggers;
	} else {
		p.expression = null;
		p.expression_enabled = false;
		p.expression_triggers_process = false;
	}
	// Runtime overlay: expression_error always; refreshed options only for a StringParam.
	p.expression_error = runtime?.expression_error ?? null;
	if (p.type === 'string' && runtime?.options !== undefined) p.options = runtime.options;
	return p;
}

/** Assemble a full render `NodeInstanceInfo` from the doc view + catalog + runtime overlay. Pure. */
export function assembleNode(
	view: NodeView,
	docParams: DocParamLeaves,
	viewers: ViewersBlob,
	catalog: NodeTypeInfo | undefined,
	runtime: RuntimeOverlay
): NodeInstanceInfo {
	// Params iterate the CATALOG structure (static per type). When the type is unknown, fall back to
	// the doc's own param keys so at least the committed values/bindings still render.
	const params: Record<string, Record<string, ParamDescriptor>> = {};
	const groupNames = catalog ? Object.keys(catalog.params) : Object.keys(docParams);
	for (const group of groupNames) {
		params[group] = {};
		const names = catalog ? Object.keys(catalog.params[group]) : Object.keys(docParams[group] ?? {});
		for (const name of names) {
			params[group][name] = mergeParam(
				catalog?.params[group]?.[name],
				docParams[group]?.[name],
				runtime.params?.[group]?.[name]
			);
		}
	}

	return {
		uid: view.uid,
		name: view.name,
		type: view.type,
		category: catalog?.category ?? '',
		doc: catalog?.doc ?? '',
		input_slots: catalog?.input_slots ?? {},
		input_multi: catalog?.input_multi,
		output_slots: catalog?.output_slots ?? {},
		params,
		pos: view.pos,
		viewers,
		membership: runtime.membership ?? null,
		error: runtime.error ?? null,
		crashed: runtime.crashed,
		restarts: runtime.restarts,
		crashExit: runtime.crashExit,
		stage: runtime.stage,
		stats: runtime.stats ?? null,
		log_endpoint: runtime.log_endpoint ?? null
	};
}
