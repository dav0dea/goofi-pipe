/** The three-way merge of a render `NodeInstanceInfo` from doc leaves, the static per-type catalog and
 * the event-sourced runtime overlay. */
import type { NodeInstanceInfo, NodeTypeInfo, NodeStage, NodeStats, NodeRuntime } from '$lib/api/control';
import type { ParamDescriptor } from '$lib/api/types';
import type { NodeView, DocParamLeaves, FacadeFace } from './graphDoc';

export type { DocParamLeaves };

export type ViewersBlob = NodeInstanceInfo['viewers'];

/** Per-param runtime overlay (event-sourced, never in the doc). */
export interface ParamRuntime {
	error?: string | null;
	/** Refreshed StringParam options (device/stream pickers) — override the catalog default. */
	options?: string[] | null;
	/** For a driven param, its LIVE evaluated value — it overrides the committed doc leaf. */
	liveValue?: unknown;
}

/** Node-level runtime overlay (event-sourced, never in the doc). */
export interface RuntimeOverlay {
	error?: string | null;
	stage?: NodeStage;
	/** Which tier the node's type currently runs on; the GIL tripwire can demote a Python type. */
	runtime?: NodeRuntime;
	stats?: NodeStats | null;
	params?: Record<string, Record<string, ParamRuntime>>;
}

/** A descriptor for a param whose node type is absent from the catalog: no bounds, no options. */
function unknownParam(): ParamDescriptor {
	return {
		type: 'unknown',
		value: undefined,
		doc: null,
		refreshable: false,
		mode: 'constant',
		expression: null,
		reference: null,
		triggers: false,
		error: null
	};
}

function mergeParam(
	catalog: ParamDescriptor | undefined,
	leaf: DocParamLeaves[string][string] | undefined,
	runtime: ParamRuntime | undefined
): ParamDescriptor {
	// Shallow-copy so the shared catalog descriptor is never mutated.
	const p: ParamDescriptor = catalog ? { ...catalog } : unknownParam();
	if (leaf && leaf.value !== undefined) (p as { value: unknown }).value = leaf.value;
	p.mode = leaf?.source?.mode ?? 'constant';
	p.expression = leaf?.source?.expr ?? null;
	p.reference = leaf?.source?.ref ?? null;
	p.triggers = leaf?.source?.triggers === true;
	p.error = runtime?.error ?? null;
	if (p.type === 'string' && runtime?.options !== undefined) p.options = runtime.options;
	if (p.type !== 'pulse' && p.mode !== 'constant' && runtime?.liveValue !== undefined)
		(p as { value: unknown }).value = runtime.liveValue;
	return p;
}

/** Assemble a full render `NodeInstanceInfo` from the doc view, catalog and runtime overlay. A
 * facade has no catalog entry — it runs nothing — so its `face` carries the slots instead. */
export function assembleNode(
	view: NodeView,
	docParams: DocParamLeaves,
	viewers: ViewersBlob,
	catalog: NodeTypeInfo | undefined,
	runtime: RuntimeOverlay,
	face?: FacadeFace
): NodeInstanceInfo {
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
		doc: catalog?.doc ?? '',
		input_slots: face?.input_slots ?? catalog?.input_slots ?? {},
		input_multi: catalog?.input_multi,
		output_slots: face?.output_slots ?? catalog?.output_slots ?? {},
		slot_labels: face?.slot_labels,
		params,
		pos: view.pos,
		viewers,
		scope: view.scope,
		error: runtime.error ?? null,
		stage: runtime.stage,
		runtime: runtime.runtime,
		stats: runtime.stats ?? null,
		subpatch: face && { memberCount: face.memberCount }
	};
}
