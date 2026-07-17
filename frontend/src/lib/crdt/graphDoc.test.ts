import { describe, it, expect } from 'vitest';
import * as Y from 'yjs';
import {
	nodesMap,
	linksArray,
	nodeView,
	nodeViews,
	paramValue,
	paramExprSource,
	linkViews
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
		expect(paramExprSource(doc, 'a', 'oscillator', 'waveform')).toBe("nd('lfo')");
		expect(paramExprSource(doc, 'a', 'common', 'max_frequency')).toBeUndefined();
	});

	it('reads links', () => {
		const doc = seedDoc();
		expect(linkViews(doc)).toEqual([
			{ node_out: 'a', slot_out: 'out', node_in: 'b', slot_in: 'data' }
		]);
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
