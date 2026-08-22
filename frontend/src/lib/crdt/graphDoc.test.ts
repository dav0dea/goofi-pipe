import { describe, it, expect } from 'vitest';
import {
	nodesMap,
	nodeView,
	nodeViews,
	setParamExpr,
	linkViews,
	instanceView,
	instanceViews,
	docParams,
	globalViews,
	isValidGlobalName,
	arrangementRoot,
	type Doc
} from './graphDoc';

/** A document in the exact shape `goofi_bridge::projection` builds. */
function seedDoc(): Doc {
	return {
		nodes: {
			a: {
				type: 'Oscillator',
				name: 'osc0',
				pos: { x: 10, y: 20 },
				params: {
					common: { max_frequency: { value: 30 } },
					oscillator: {
						waveform: { value: 'sine', expr: { source: "nd('lfo')", enabled: true, triggers: false } }
					}
				}
			},
			b: { type: 'Buffer', name: 'buf0' }
		},
		links: [{ node_out: 'a', slot_out: 'out', node_in: 'b', slot_in: 'data' }],
		instances: {},
		globals: {},
		arrangement: {}
	};
}

describe('graphDoc readers', () => {
	it('reads node identity views', () => {
		const doc = seedDoc();
		expect(nodeView(doc, 'a')).toEqual({ uid: 'a', type: 'Oscillator', name: 'osc0', pos: [10, 20] });
		// A node with no pos defaults to [0,0].
		expect(nodeView(doc, 'b')).toEqual({ uid: 'b', type: 'Buffer', name: 'buf0', pos: [0, 0] });
		expect(nodeView(doc, 'missing')).toBeNull();
		expect(nodeViews(doc).map((n) => n.uid)).toEqual(['a', 'b']);
	});

	it('reads param values and expression sources', () => {
		const doc = seedDoc();
		expect(docParams(doc, 'a').common?.max_frequency?.value).toBe(30);
		expect(docParams(doc, 'a').oscillator?.waveform?.value).toBe('sine');
		expect(docParams(doc, 'a').common?.nope?.value).toBeUndefined();
		expect(docParams(doc, 'a').oscillator?.waveform?.expr?.source).toBe("nd('lfo')");
		expect(docParams(doc, 'a').common?.max_frequency?.expr?.source).toBeUndefined();
	});

	it('reads links', () => {
		const doc = seedDoc();
		expect(linkViews(doc)).toEqual([
			{ node_out: 'a', slot_out: 'out', node_in: 'b', slot_in: 'data' }
		]);
	});

	it('reads the sub-patch forest (scopes, members, stubs)', () => {
		// The flat shape the projection writes: members keyed by uid → {is_instance}, stubs by id.
		const doc: Doc = {
			...seedDoc(),
			instances: {
				i1: {
					name: 'subpatch0',
					parent: '__root__',
					pos: { x: 5, y: 6 },
					members: { m1: { is_instance: false } },
					stubs: {
						out0: {
							dir: 'out',
							dtype: 'ARRAY',
							name: 'wave',
							pos: { x: 1, y: 2 },
							inner_node: 'm1',
							inner_slot: 'out'
						}
					}
				}
			}
		};

		expect(instanceViews(doc).map((i) => i.uid)).toEqual(['i1']);
		expect(instanceView(doc, 'i1')).toEqual({
			uid: 'i1',
			name: 'subpatch0',
			parent: '__root__',
			pos: [5, 6],
			members: { m1: false },
			interface: [
				{
					bnd_id: 'out0',
					dir: 'out',
					dtype: 'ARRAY',
					name: 'wave',
					pos: [1, 2],
					inner_node: 'm1',
					inner_slot: 'out'
				}
			]
		});
		expect(instanceView(doc, 'missing')).toBeNull();
	});

	it('reads a node param leaves (value + expression binding) via docParams', () => {
		const doc = seedDoc();
		const p = docParams(doc, 'a');
		expect(p.common.max_frequency).toEqual({ value: 30 });
		// waveform in seedDoc carries a value AND an expr binding.
		expect(p.oscillator.waveform).toEqual({
			value: 'sine',
			expr: { source: "nd('lfo')", enabled: true, triggers: false }
		});
		// A node with no params → empty.
		expect(docParams(doc, 'b')).toEqual({});
	});

	it('an unwired stub omits its inner pair rather than carrying an empty one', () => {
		// The projection PRUNES `inner_node`/`inner_slot` when a stub is unwired, and a merge patch
		// deletes them with a null. A reader that answered `''` would draw a cable to nowhere.
		const doc: Doc = {
			...seedDoc(),
			instances: { i1: { name: 's', parent: '__root__', stubs: { in0: { dir: 'in', dtype: 'ARRAY', name: 'a' } } } }
		};
		const stub = instanceView(doc, 'i1')!.interface[0];
		expect(stub.inner_node).toBeUndefined();
		expect(stub.inner_slot).toBeUndefined();
	});

	it('a wrongly-typed or absent root reads as empty rather than throwing', () => {
		// The manager is the sole author, so this can only mean the two ends have drifted — and a
		// half-drawn graph reports that better than a blank page does.
		expect(nodeViews({})).toEqual([]);
		expect(linkViews({ links: 'not an array' })).toEqual([]);
		expect(globalViews({ globals: null })).toEqual([]);
		expect(instanceViews({ instances: 7 })).toEqual([]);
	});
});

describe('graphDoc.setParamExpr — the test-seed binding write', () => {
	it('writes a binding in place and docParams reads it back', () => {
		const doc = seedDoc();
		expect(
			setParamExpr(doc, 'a', 'common', 'max_frequency', { source: "nd('f')", enabled: true, triggers: false })
		).toBe(true);
		expect(docParams(doc, 'a').common?.max_frequency?.expr).toEqual({
			source: "nd('f')",
			enabled: true,
			triggers: false
		});
		// The committed value is untouched — only the binding was written.
		expect(docParams(doc, 'a').common?.max_frequency?.value).toBe(30);
	});

	it('clears a binding when passed null', () => {
		const doc = seedDoc();
		// `waveform` is seeded WITH an expr in seedDoc.
		expect(docParams(doc, 'a').oscillator?.waveform?.expr).toBeDefined();
		expect(setParamExpr(doc, 'a', 'oscillator', 'waveform', null)).toBe(true);
		expect(docParams(doc, 'a').oscillator?.waveform?.expr).toBeUndefined();
	});

	it('no-ops (returns false) when the node is absent — never mint a phantom', () => {
		const doc = seedDoc();
		expect(setParamExpr(doc, 'ghost', 'common', 'x', { source: 'nd()', enabled: true, triggers: false })).toBe(false);
		expect(nodesMap(doc).ghost).toBeUndefined();
	});
});

describe('graphDoc globals', () => {
	it('reads global views (system-first, typed, with the system flag)', () => {
		const doc: Doc = {
			...seedDoc(),
			globals: {
				default_ufreq: { value: 30, type: 'float', system: true },
				subject: { value: 'P07', type: 'string', system: false }
			}
		};
		expect(globalViews(doc)).toEqual([
			{ name: 'default_ufreq', value: 30, type: 'float', system: true },
			{ name: 'subject', value: 'P07', type: 'string', system: false }
		]);
	});

	it('validates global names like the Rust identifier rule', () => {
		expect(isValidGlobalName('default_ufreq')).toBe(true);
		expect(isValidGlobalName('_x1')).toBe(true);
		expect(isValidGlobalName('')).toBe(false);
		expect(isValidGlobalName('1x')).toBe(false);
		expect(isValidGlobalName('a b')).toBe(false);
		expect(isValidGlobalName('a.b')).toBe(false);
		expect(isValidGlobalName('globals')).toBe(false);
	});

	/* The arrangement parser. It reads a tree straight into the shape the panel system draws, so it
	   is the one place a malformed document could put a hole on screen — and it is fed by the wire,
	   which means anything it cannot make sense of has to be DROPPED rather than half-drawn. */
	it('reads the arrangement into the tree the panel system draws', () => {
		const doc = {
			arrangement: {
				'#seq': 4,
				root: {
					kind: 'stack',
					id: 'stack-1',
					children: [
						{
							kind: 'split',
							id: 'split-4',
							axis: 'column',
							size: 1.0,
							children: [
								{ kind: 'panel', id: 'panel-2', size: 0.6, panel_type: 'node-editor', state: 'null' },
								{
									kind: 'panel',
									id: 'panel-3',
									size: 0.4,
									panel_type: 'viewer',
									state: '{"node":"a1b2","kind":"line"}'
								}
							]
						},
						{ kind: 'panel', id: 'panel-7', size: 1.0, panel_type: 'console', state: 'null' }
					]
				}
			}
		};
		expect(arrangementRoot(doc)).toEqual({
			kind: 'stack',
			id: 'stack-1',
			children: [
				{
					kind: 'split',
					id: 'split-4',
					direction: 'column',
					// A child's share rides ON the child, and the renderer wants them per split — so
					// they are collected on the way up rather than held twice.
					sizes: [0.6, 0.4],
					children: [
						{ kind: 'panel', id: 'panel-2', panelType: 'node-editor', state: undefined },
						{ kind: 'panel', id: 'panel-3', panelType: 'viewer', state: { node: 'a1b2', kind: 'line' } }
					]
				},
				{ kind: 'panel', id: 'panel-7', panelType: 'console', state: undefined }
			]
		});
	});

	it('drops what it cannot draw instead of drawing a hole', () => {
		// A root that will not parse, a container with no children, a node with no id, and a state
		// leaf that is not the JSON STRING the wire promises — each is a shape the manager never
		// writes, and each would otherwise reach the renderer as a gap. A root that is not a tab
		// GROUP is dropped too: the page strip is what it draws.
		expect(arrangementRoot({ arrangement: {} })).toBeNull();
		expect(arrangementRoot({ arrangement: { root: 'nope' } })).toBeNull();
		expect(arrangementRoot({ arrangement: { root: { kind: 'stack', children: [] } } })).toBeNull();
		expect(
			arrangementRoot({ arrangement: { root: { kind: 'stack', id: 's', children: [] } } })
		).toBeNull();
		expect(
			arrangementRoot({ arrangement: { root: { kind: 'panel', id: 'p', panel_type: 'console' } } }),
			'a lone panel is not a page strip'
		).toBeNull();
		expect(
			arrangementRoot({
				arrangement: {
					root: {
						kind: 'stack',
						id: 's',
						children: [{ kind: 'panel', id: 'p', state: { node: 'x' } }, { kind: 'split', id: 'x', children: [] }]
					}
				}
			})
		).toEqual({
			kind: 'stack',
			id: 's',
			children: [{ kind: 'panel', id: 'p', panelType: 'empty', state: undefined }]
		});
	});
});
