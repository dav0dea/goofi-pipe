import { describe, it, expect } from 'vitest';
import {
	nodesMap,
	nodeView,
	nodeViews,
	setParamExpr,
	linkViews,
	facadeFaces,
	docParams,
	globalViews,
	isValidIdentifier,
	arrangementTabs,
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
		globals: {},
		arrangement: {}
	};
}

describe('graphDoc readers', () => {
	it('reads node identity views', () => {
		const doc = seedDoc();
		expect(nodeView(doc, 'a')).toEqual({
			uid: 'a', type: 'Oscillator', name: 'osc0', pos: [10, 20], scope: '__root__'
		});
		// A node with no pos defaults to [0,0], and one naming no scope is at the top level.
		expect(nodeView(doc, 'b')).toEqual({
			uid: 'b', type: 'Buffer', name: 'buf0', pos: [0, 0], scope: '__root__'
		});
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

	it('reads the sub-patch forest out of the one node map it is written in', () => {
		// The shape the projection writes: a facade and a port are node records, membership rides
		// each record as `scope`, and a port's inner wire is a link like any other.
		const base = seedDoc();
		const doc: Doc = {
			...base,
			nodes: {
				...(base.nodes as Record<string, unknown>),
				m1: { type: 'Buffer', name: 'm0', scope: 'i1' },
				i1: { type: 'SubPatch', name: 'subpatch0', pos: { x: 5, y: 6 } },
				p1: { type: 'OutArray', name: 'wave', pos: { x: 1, y: 2 }, scope: 'i1' }
			},
			links: [
				...(base.links as unknown[]),
				{ node_out: 'm1', slot_out: 'out', node_in: 'p1', slot_in: 'value' }
			]
		};

		// ONE list, because the document is one map: leaf, facade and port alike, each carrying the
		// scope it is drawn in.
		expect(nodeViews(doc).map((n) => [n.uid, n.scope])).toEqual([
			['a', '__root__'],
			['b', '__root__'],
			['m1', 'i1'],
			['i1', '__root__'],
			['p1', 'i1']
		]);
		// A port IS the facade's slot: keyed by the port's stable uid, labelled with its name.
		expect(facadeFaces(doc).get('i1')).toEqual({
			input_slots: {},
			output_slots: { p1: 'ARRAY' },
			slot_labels: { p1: 'wave' },
			memberCount: 2
		});
		// A uid that names a LEAF has no face, however much it looks like a scope from outside.
		expect(facadeFaces(doc).has('a')).toBe(false);
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

	it('an unwired port is a slot like any other — a facade with nothing behind it still has a face', () => {
		// The unwired state is the test: a port with no link naming it is present, named and
		// addressable, exactly as a leaf that was never connected is.
		const doc: Doc = {
			...seedDoc(),
			nodes: { i1: { type: 'SubPatch', name: 's' }, p1: { type: 'InArray', name: 'a', scope: 'i1' } },
			links: []
		};
		expect(facadeFaces(doc).get('i1')).toEqual({
			input_slots: { p1: 'ARRAY' },
			output_slots: {},
			slot_labels: { p1: 'a' },
			memberCount: 1
		});
		// …and an EMPTY sub-patch is a node too, with a face that simply exposes nothing.
		expect(facadeFaces({ nodes: { i1: { type: 'SubPatch', name: 's' } } }).get('i1')).toEqual({
			input_slots: {},
			output_slots: {},
			slot_labels: {},
			memberCount: 0
		});
	});

	it('a wrongly-typed or absent root reads as empty rather than throwing', () => {
		// The manager is the sole author, so this can only mean the two ends have drifted — and a
		// half-drawn graph reports that better than a blank page does.
		expect(nodeViews({})).toEqual([]);
		expect(linkViews({ links: 'not an array' })).toEqual([]);
		expect(globalViews({ globals: null })).toEqual([]);
		expect(facadeFaces({ nodes: 7 }).size).toBe(0);
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

	it('validates names like the Rust identifier rule', () => {
		expect(isValidIdentifier('default_ufreq')).toBe(true);
		expect(isValidIdentifier('_x1')).toBe(true);
		expect(isValidIdentifier('')).toBe(false);
		expect(isValidIdentifier('1x')).toBe(false);
		expect(isValidIdentifier('a b')).toBe(false);
		expect(isValidIdentifier('a.b')).toBe(false);
		// A keyword passes the character rule and fails the parser, and every namespace is read as
		// an ATTRIBUTE in an expression — `globals.gain`, `nd('chain').drain` — so all refuse one.
		expect(isValidIdentifier('drain')).toBe(true);
		expect(isValidIdentifier('class')).toBe(false);
		expect(isValidIdentifier('None')).toBe(false);
		expect(isValidIdentifier('nd()')).toBe(false);
		expect(isValidIdentifier("it's")).toBe(false);
		expect(isValidIdentifier('lambda')).toBe(false);
		// `globals` is goofi's own namespace token, reserved for globals AND node names alike.
		expect(isValidIdentifier('globals')).toBe(false);
	});

	/* The arrangement parser. It reads a tree straight into the shape the panel system draws, so it
	   is the one place a malformed document could put a hole on screen — and it is fed by the wire,
	   which means anything it cannot make sense of has to be DROPPED rather than half-drawn. */
	it('reads the tab strip into the tree the panel system draws', () => {
		const doc = {
			arrangement: {
				'#seq': 4,
				tabs: [
					{
						id: 'tab-1',
						name: 'Tab 1',
						root: {
							kind: 'split',
							id: 'split-4',
							axis: 'column',
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
						}
					}
				]
			}
		};
		expect(arrangementTabs(doc)).toEqual([
			{
				id: 'tab-1',
				name: 'Tab 1',
				root: {
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
				}
			}
		]);
	});

	it('drops what it cannot draw instead of drawing a hole', () => {
		// A tab whose root will not parse, a split with no children, a node with no id, and a state
		// leaf that is not the JSON STRING the wire promises — each is a shape the manager never
		// writes, and each would otherwise reach the renderer as a gap.
		const tab = (root: unknown): unknown => ({ id: 't', name: 'T', root });
		expect(arrangementTabs({ arrangement: {} })).toEqual([]);
		expect(arrangementTabs({ arrangement: { tabs: 'nope' } })).toEqual([]);
		expect(arrangementTabs({ arrangement: { tabs: [tab({ kind: 'panel' })] } })).toEqual([]);
		expect(
			arrangementTabs({ arrangement: { tabs: [tab({ kind: 'split', id: 's', children: [] })] } })
		).toEqual([]);
		expect(
			arrangementTabs({ arrangement: { tabs: [tab({ kind: 'panel', id: 'p', state: { node: 'x' } })] } })
		).toEqual([
			{ id: 't', name: 'T', root: { kind: 'panel', id: 'p', panelType: 'empty', state: undefined } }
		]);
	});

	it('gives a root the whole tab, whatever the wire says about its share', () => {
		// A root carries no share on the wire — it fills its tab — so the parser must not read one.
		const [ws] = arrangementTabs({
			arrangement: { tabs: [{ id: 't', name: 'T', root: { kind: 'panel', id: 'p', panel_type: 'console' } }] }
		});
		expect(ws.root).toEqual({ kind: 'panel', id: 'p', panelType: 'console', state: undefined });
	});
});
