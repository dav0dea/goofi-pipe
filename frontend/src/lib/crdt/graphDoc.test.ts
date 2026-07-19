import { describe, it, expect } from 'vitest';
import * as Y from 'yjs';
import {
	nodesMap,
	linksArray,
	nodeView,
	nodeViews,
	paramValue,
	setParamExpr,
	linkViews,
	instancesMap,
	instanceView,
	instanceViews,
	setNodePos,
	setViewers,
	viewersJson,
	docParams,
	globalsMap,
	globalViews,
	isValidGlobalName,
	setGlobalValue,
	addGlobal,
	removeGlobal,
	renameGlobal
} from './graphDoc';

/** Build a doc in the exact shape the Rust `GraphDoc` mirror writes. */
function seedDoc(): Y.Doc {
	const doc = new Y.Doc();
	const nodes = nodesMap(doc);

	const osc = new Y.Map<unknown>();
	osc.set('type', 'Oscillator');
	osc.set('name', 'osc0');
	const pos = new Y.Map<unknown>();
	pos.set('x', 10);
	pos.set('y', 20);
	osc.set('pos', pos);
	const params = new Y.Map<unknown>();
	const common = new Y.Map<unknown>();
	const maxFreq = new Y.Map<unknown>();
	maxFreq.set('value', 30);
	common.set('max_frequency', maxFreq);
	const oscGroup = new Y.Map<unknown>();
	const waveform = new Y.Map<unknown>();
	waveform.set('value', 'sine');
	const expr = new Y.Map<unknown>();
	expr.set('source', "nd('lfo')");
	expr.set('enabled', true);
	expr.set('triggers', false);
	waveform.set('expr', expr);
	oscGroup.set('waveform', waveform);
	params.set('common', common);
	params.set('oscillator', oscGroup);
	osc.set('params', params);
	nodes.set('a', osc);

	const buf = new Y.Map<unknown>();
	buf.set('type', 'Buffer');
	buf.set('name', 'buf0');
	nodes.set('b', buf);

	const link = new Y.Map<unknown>();
	link.set('node_out', 'a');
	link.set('slot_out', 'out');
	link.set('node_in', 'b');
	link.set('slot_in', 'data');
	linksArray(doc).push([link]);

	return doc;
}

describe('graphDoc readers', () => {
	it('reads node identity views', () => {
		const doc = seedDoc();
		expect(nodeView(doc, 'a')).toEqual({ uid: 'a', type: 'Oscillator', name: 'osc0', pos: [10, 20] });
		// A node with no pos map defaults to [0,0].
		expect(nodeView(doc, 'b')).toEqual({ uid: 'b', type: 'Buffer', name: 'buf0', pos: [0, 0] });
		expect(nodeView(doc, 'missing')).toBeNull();
		expect(nodeViews(doc).map((n) => n.uid)).toEqual(['a', 'b']);
	});

	it('reads param values and expression sources', () => {
		const doc = seedDoc();
		expect(paramValue(doc, 'a', 'common', 'max_frequency')).toBe(30);
		expect(paramValue(doc, 'a', 'oscillator', 'waveform')).toBe('sine');
		expect(paramValue(doc, 'a', 'common', 'nope')).toBeUndefined();
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
		// Build a scope in the exact flat shape the Rust mirror writes: members keyed by uid →
		// {is_instance}, stubs keyed by id.
		const doc = new Y.Doc();
		const inst = new Y.Map<unknown>();
		inst.set('name', 'subpatch0');
		inst.set('parent', '__root__');
		const ipos = new Y.Map<unknown>();
		ipos.set('x', 5);
		ipos.set('y', 6);
		inst.set('pos', ipos);
		const members = new Y.Map<unknown>();
		const m1 = new Y.Map<unknown>();
		m1.set('is_instance', false);
		members.set('m1', m1);
		inst.set('members', members);
		const stubs = new Y.Map<unknown>();
		const out0 = new Y.Map<unknown>();
		out0.set('dir', 'out');
		out0.set('dtype', 'ARRAY');
		out0.set('name', 'wave');
		const bpos = new Y.Map<unknown>();
		bpos.set('x', 1);
		bpos.set('y', 2);
		out0.set('pos', bpos);
		out0.set('inner_node', 'm1');
		out0.set('inner_slot', 'out');
		stubs.set('out0', out0);
		inst.set('stubs', stubs);
		instancesMap(doc).set('i1', inst);

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

	it('reflects live edits (the reactive-read contract)', () => {
		const doc = seedDoc();
		(nodesMap(doc).get('a')!.get('params') as Y.Map<Y.Map<Y.Map<unknown>>>)
			.get('common')!
			.get('max_frequency')!
			.set('value', 42);
		expect(paramValue(doc, 'a', 'common', 'max_frequency')).toBe(42);
	});
});

describe('graphDoc.setNodePos — committed-drag leaf write', () => {
	it('writes a node position in place and the reader reflects it', () => {
		const doc = seedDoc();
		expect(setNodePos(doc, 'a', [111, 222])).toBe(true);
		expect(nodeView(doc, 'a')!.pos).toEqual([111, 222]);
		// Identity fields are untouched — only pos moved.
		expect(nodeView(doc, 'a')!.type).toBe('Oscillator');
	});

	it('writes a sub-patch instance box position in place', () => {
		const doc = new Y.Doc();
		const inst = new Y.Map<unknown>();
		inst.set('name', 'subpatch0');
		inst.set('parent', '__root__');
		instancesMap(doc).set('i1', inst);

		expect(setNodePos(doc, 'i1', [30, 40])).toBe(true);
		expect(instanceView(doc, 'i1')!.pos).toEqual([30, 40]);
	});

	it('no-ops (returns false) when the uid is in neither map — never mint a phantom', () => {
		const doc = seedDoc();
		expect(setNodePos(doc, 'ghost', [1, 2])).toBe(false);
		expect(nodesMap(doc).get('ghost')).toBeUndefined();
		expect(instancesMap(doc).get('ghost')).toBeUndefined();
	});

	it('is idempotent — re-writing the current position creates no new struct', () => {
		const doc = seedDoc();
		setNodePos(doc, 'a', [111, 222]);
		// The state vector counts each client's struct clock; a real write bumps it, a guarded
		// no-op leaves it untouched. (An update payload can't be used — it always carries the
		// doc's full delete set, so it's non-empty once any value has been overwritten.)
		const sv = Y.encodeStateVector(doc);
		setNodePos(doc, 'a', [111, 222]);
		expect(Y.encodeStateVector(doc)).toEqual(sv);
	});
});

describe('graphDoc.setParamExpr — expression-binding leaf write', () => {
	it('writes a binding in place and paramExpr reads it back', () => {
		const doc = seedDoc();
		expect(setParamExpr(doc, 'a', 'common', 'max_frequency', { source: "nd('f')", enabled: true, triggers: false })).toBe(
			true
		);
		expect(docParams(doc, 'a').common?.max_frequency?.expr).toEqual({
			source: "nd('f')",
			enabled: true,
			triggers: false
		});
		// The committed value is untouched — only the binding was written.
		expect(paramValue(doc, 'a', 'common', 'max_frequency')).toBe(30);
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
		expect(nodesMap(doc).get('ghost')).toBeUndefined();
	});

	it('is idempotent — re-writing the same binding creates no new struct', () => {
		const doc = seedDoc();
		const b = { source: "nd('lfo')", enabled: true, triggers: true };
		setParamExpr(doc, 'a', 'common', 'max_frequency', b);
		const sv = Y.encodeStateVector(doc);
		setParamExpr(doc, 'a', 'common', 'max_frequency', b);
		expect(Y.encodeStateVector(doc)).toEqual(sv);
	});
});

describe('graphDoc.setViewers — view-state leaf write', () => {
	it('writes a node viewer blob and reads it back', () => {
		const doc = seedDoc();
		const blob = { out: { collapsed: false, kind: 'line', settings: {} } };
		expect(setViewers(doc, 'a', blob)).toBe(true);
		expect(viewersJson(doc, 'a')).toEqual(blob);
	});

	it('no-ops (returns false) when the node is absent — never mint a phantom', () => {
		const doc = seedDoc();
		expect(setViewers(doc, 'ghost', { out: {} })).toBe(false);
		expect(nodesMap(doc).get('ghost')).toBeUndefined();
	});

	it('is idempotent — re-writing the same blob creates no new struct', () => {
		const doc = seedDoc();
		const blob = { out: { collapsed: true, kind: 'spectrum', settings: { db: -60 } } };
		setViewers(doc, 'a', blob);
		const sv = Y.encodeStateVector(doc);
		setViewers(doc, 'a', blob);
		expect(Y.encodeStateVector(doc)).toEqual(sv);
	});
});

/** Seed a globals root in the exact `{value, type, system}` shape the Rust mirror writes. */
function seedGlobals(doc: Y.Doc): void {
	const g = globalsMap(doc);
	const uf = new Y.Map<unknown>();
	uf.set('value', 30);
	uf.set('type', 'float');
	uf.set('system', true);
	g.set('default_ufreq', uf);
	const subj = new Y.Map<unknown>();
	subj.set('value', 'P07');
	subj.set('type', 'string');
	subj.set('system', false);
	g.set('subject', subj);
}

describe('graphDoc globals', () => {
	it('reads global views (system-first, typed, with the system flag)', () => {
		const doc = new Y.Doc();
		seedGlobals(doc);
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

	it('edits an existing value in place, keeping type + system', () => {
		const doc = new Y.Doc();
		seedGlobals(doc);
		expect(setGlobalValue(doc, 'default_ufreq', 45)).toBe(true);
		expect(globalViews(doc)[0]).toEqual({ name: 'default_ufreq', value: 45, type: 'float', system: true });
		// Absent global → no-op (never mint via this path).
		expect(setGlobalValue(doc, 'ghost', 1)).toBe(false);
	});

	it('adds a new user global, refusing collisions + invalid names', () => {
		const doc = new Y.Doc();
		seedGlobals(doc);
		expect(addGlobal(doc, 'gain', 2.5, 'float')).toBe(true);
		expect(globalViews(doc).find((v) => v.name === 'gain')).toEqual({
			name: 'gain',
			value: 2.5,
			type: 'float',
			system: false
		});
		expect(addGlobal(doc, 'default_ufreq', 1, 'float')).toBe(false); // collision (system)
		expect(addGlobal(doc, 'subject', 'x', 'string')).toBe(false); // collision (user)
		expect(addGlobal(doc, '1bad', 0, 'int')).toBe(false); // invalid name
	});

	it('removes a user global but refuses a system one', () => {
		const doc = new Y.Doc();
		seedGlobals(doc);
		expect(removeGlobal(doc, 'subject')).toBe(true);
		expect(globalsMap(doc).has('subject')).toBe(false);
		expect(removeGlobal(doc, 'default_ufreq')).toBe(false); // system, protected
		expect(globalsMap(doc).has('default_ufreq')).toBe(true);
	});

	it('renames a user global (delete+add, carrying value+type), refusing bad cases', () => {
		const doc = new Y.Doc();
		seedGlobals(doc);
		addGlobal(doc, 'gain', 2.0, 'float');
		expect(renameGlobal(doc, 'gain', 'gain_a')).toBe(true);
		expect(globalsMap(doc).has('gain')).toBe(false);
		expect(globalViews(doc).find((v) => v.name === 'gain_a')).toEqual({
			name: 'gain_a',
			value: 2.0,
			type: 'float',
			system: false
		});
		expect(renameGlobal(doc, 'default_ufreq', 'x')).toBe(false); // system
		expect(renameGlobal(doc, 'gain_a', 'subject')).toBe(false); // collision
		expect(renameGlobal(doc, 'gain_a', '1bad')).toBe(false); // invalid
	});
});
