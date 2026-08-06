import { describe, it, expect } from 'vitest';
import { assembleNode, type DocParamLeaves, type RuntimeOverlay } from './nodeAssembly';
import type { NodeTypeInfo } from '$lib/api/control';
import type { NodeView } from './graphDoc';

const view: NodeView = { uid: 'n1', type: 'Oscillator', name: 'osc0', pos: [10, 20] };

/** A catalog entry with a static float param + a refreshable string param. */
function catalog(): NodeTypeInfo {
	return {
		type: 'Oscillator',
		category: 'inputs',
		doc: 'A sine generator',
		available: true,
		missing_deps: [],
		input_slots: { in: 'ARRAY' },
		input_multi: ['in'],
		output_slots: { out: 'ARRAY' },
		params: {
			common: {
				frequency: {
					type: 'float',
					value: 1,
					vmin: 0,
					vmax: 30,
					doc: null,
					refreshable: false,
					expression: null,
					expression_enabled: false,
					expression_triggers_process: false,
					expression_error: null
				}
			},
			audio: {
				device: {
					type: 'string',
					value: 'default',
					options: ['default'],
					doc: null,
					refreshable: true,
					expression: null,
					expression_enabled: false,
					expression_triggers_process: false,
					expression_error: null
				}
			}
		}
	};
}

describe('assembleNode — three-way merge', () => {
	it('merges catalog structure + doc values + runtime into a full node', () => {
		const docParams: DocParamLeaves = { common: { frequency: { value: 12 } } };
		const runtime: RuntimeOverlay = { error: null, stage: 'ready' };
		const n = assembleNode(view, docParams, {}, catalog(), runtime);

		// Identity + pos from the doc view.
		expect([n.uid, n.type, n.name, n.pos]).toEqual(['n1', 'Oscillator', 'osc0', [10, 20]]);
		// Catalog supplies category/doc/slots and the param STRUCTURE.
		expect(n.category).toBe('inputs');
		expect(n.input_slots).toEqual({ in: 'ARRAY' });
		expect(n.input_multi).toEqual(['in']);
		expect(n.output_slots).toEqual({ out: 'ARRAY' });
		const freq = n.params.common.frequency;
		expect(freq.type).toBe('float');
		expect((freq as { vmin: number }).vmin).toBe(0);
		expect((freq as { vmax: number }).vmax).toBe(30);
		// Doc overrides the value; a param with no doc leaf keeps the catalog default.
		expect(freq.value).toBe(12);
		expect(n.params.audio.device.value).toBe('default');
		// Runtime supplies stage/error.
		expect(n.stage).toBe('ready');
		expect(n.error).toBeNull();
	});

	it('does not mutate the shared catalog descriptor', () => {
		const cat = catalog();
		const docParams: DocParamLeaves = { common: { frequency: { value: 99 } } };
		assembleNode(view, docParams, {}, cat, {});
		// The catalog's default value must be untouched by the per-node merge.
		expect(cat.params.common.frequency.value).toBe(1);
	});

	it('applies a doc expression binding + the runtime expression_error', () => {
		const docParams: DocParamLeaves = {
			common: { frequency: { value: 1, expr: { source: "nd('lfo')", enabled: true, triggers: true } } }
		};
		const runtime: RuntimeOverlay = {
			params: { common: { frequency: { expression_error: 'name error: lfo' } } }
		};
		const freq = assembleNode(view, docParams, {}, catalog(), runtime).params.common.frequency;
		expect(freq.expression).toBe("nd('lfo')");
		expect(freq.expression_enabled).toBe(true);
		expect(freq.expression_triggers_process).toBe(true);
		expect(freq.expression_error).toBe('name error: lfo');
	});

	it('clears a stale binding when the doc leaf has no expr', () => {
		// The catalog default carries no binding; a doc leaf without `expr` must leave it cleared
		// (never inherit a binding the doc doesn't have).
		const freq = assembleNode(view, { common: { frequency: { value: 5 } } }, {}, catalog(), {}).params
			.common.frequency;
		expect(freq.expression).toBeNull();
		expect(freq.expression_enabled).toBe(false);
	});

	it('overrides StringParam options from runtime (a refresh), not the catalog default', () => {
		const runtime: RuntimeOverlay = {
			params: { audio: { device: { options: ['default', 'HD Audio', 'USB Mic'] } } }
		};
		const device = assembleNode(view, {}, {}, catalog(), runtime).params.audio.device;
		expect((device as { options: string[] }).options).toEqual(['default', 'HD Audio', 'USB Mic']);
	});

	it('falls back to an empty-descriptor node when the type is not in the catalog', () => {
		// A missing-deps type: no catalog entry. Identity/pos/error still render; params come from the
		// doc keys as unknown descriptors so committed values are not lost.
		const docParams: DocParamLeaves = { common: { frequency: { value: 7 } } };
		const runtime: RuntimeOverlay = { error: 'missing dep: numpy', stage: 'error' };
		const n = assembleNode(view, docParams, {}, undefined, runtime);
		expect(n.name).toBe('osc0');
		expect(n.pos).toEqual([10, 20]);
		expect(n.input_slots).toEqual({});
		expect(n.output_slots).toEqual({});
		expect(n.category).toBe('');
		expect(n.params.common.frequency.type).toBe('unknown');
		expect(n.params.common.frequency.value).toBe(7);
		expect(n.error).toBe('missing dep: numpy');
		expect(n.stage).toBe('error');
	});

	it('passes the viewers blob through verbatim', () => {
		const viewers = { out: { collapsed: false, kind: 'line', settings: {} } };
		expect(assembleNode(view, {}, viewers, catalog(), {}).viewers).toEqual(viewers);
	});
});
