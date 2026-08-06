import { describe, it, expect } from 'vitest';
import { NODE, inputPorts } from './nodeMetrics';
import { inputAnchors, nearSlots, sameKeys, SLOT_PROXIMITY_PX, type AnchorNode } from './slotProximity';

/*
 * The arithmetic behind the input-name proximity reveal.
 *
 * An input slot's name is drawn nowhere else in the app — the `.conn-label` on its connector pill is
 * the whole rendering — and a mouse asks for it by hovering the pill. A finger cannot, so R rested
 * every one of them open under a coarse pointer, which put a name tag on every input on the canvas
 * at all times. The reveal is by PROXIMITY now, while a cable is in flight: the inputs the pointer
 * is closing on name themselves, for both modalities, and nothing names itself the rest of the time.
 *
 * All of it lives here, DOM-free, because the component that drives it cannot be mounted in this
 * repo's vitest — the same split `longPress.ts`, `eventPoint.ts` and `doubleTapZoom.ts` are built on.
 */

const key = (uid: string, slot: string): string => `${uid}|${slot}`;

describe('SLOT_PROXIMITY_PX', () => {
	/* One coarse `--hit` (app.css raises `--hit` to 44px under the coarse idiom).
	 *
	 * The radius answers "which inputs is this drop choosing between", and a fingertip target IS that
	 * neighbourhood — so the tags that light up are exactly the ones a finger could plausibly land
	 * on. Input connectors tile the left edge at `--node-u` (24px), so 44px reaches roughly two slots
	 * either side of the pointer: enough to tell them apart, far short of naming a whole node.
	 * Measured in SCREEN px, not flow px, because a fingertip is a physical size — the caller divides
	 * by the live zoom before comparing against flow-space anchors. */
	it('is one coarse --hit', () => {
		expect(SLOT_PROXIMITY_PX).toBe(44);
	});
});

describe('nearSlots', () => {
	const anchors = [
		{ key: 'a', x: 0, y: 0 },
		{ key: 'b', x: 100, y: 0 }
	];

	it('takes an anchor inside the radius', () => {
		expect([...nearSlots(anchors, { x: 6, y: 8 }, 20)]).toEqual(['a']);
	});

	it('leaves one outside it alone', () => {
		expect([...nearSlots(anchors, { x: 60, y: 0 }, 20)]).toEqual([]);
	});

	/* Inclusive: the radius names how far the reveal reaches, so a point exactly at it is within
	   reach. A strict `<` would make the boundary a coin flip on floating-point noise. */
	it('takes one exactly ON the boundary', () => {
		expect([...nearSlots(anchors, { x: 12, y: 16 }, 20)], 'a 3-4-5 triangle × 4').toEqual(['a']);
		expect([...nearSlots(anchors, { x: 12.001, y: 16 }, 20)], 'and one hair past it').toEqual([]);
	});

	it('takes every anchor in range, and only those', () => {
		const many = [
			{ key: 'a', x: 0, y: 0 },
			{ key: 'b', x: 0, y: 24 },
			{ key: 'c', x: 0, y: 48 },
			{ key: 'd', x: 0, y: 200 }
		];
		expect([...nearSlots(many, { x: 0, y: 24 }, 25)].sort()).toEqual(['a', 'b', 'c']);
	});

	it('answers empty for no anchors, and for a zero radius off the anchor', () => {
		expect(nearSlots([], { x: 0, y: 0 }, 44).size).toBe(0);
		expect(nearSlots(anchors, { x: 1, y: 0 }, 0).size).toBe(0);
		expect([...nearSlots(anchors, { x: 0, y: 0 }, 0)], 'dead on, at zero radius').toEqual(['a']);
	});
});

describe('sameKeys', () => {
	/* The publisher writes a new Set on every pointermove; without this every move would invalidate
	   every node on the canvas for a set that usually did not change. */
	it('is true only for the same membership', () => {
		expect(sameKeys(new Set(['a', 'b']), new Set(['b', 'a']))).toBe(true);
		expect(sameKeys(new Set(), new Set())).toBe(true);
		expect(sameKeys(new Set(['a']), new Set(['a', 'b'])), 'a different size').toBe(false);
		expect(sameKeys(new Set(['a']), new Set(['b'])), 'the same size, other members').toBe(false);
	});
});

describe('inputAnchors', () => {
	const node = (over: Partial<AnchorNode> = {}): AnchorNode => ({
		uid: 'n1',
		x: 10,
		y: 20,
		slots: ['in', 'other'],
		multi: new Set<string>(),
		...over
	});

	/* The anchor is the connector's own centre: `.conn.in` hugs the node's LEFT edge (x = the node
	   position) at the `top` nodeMetrics already computes for the cable anchors — so this reveal and
	   the cable it reveals for are measured from the same point. */
	it('puts each input on the node’s left edge, at nodeMetrics’ own pitch', () => {
		const ports = inputPorts(['in', 'other'], () => false);
		expect(inputAnchors([node()], key)).toEqual([
			{ key: 'n1|in', x: 10, y: 20 + ports[0].top },
			{ key: 'n1|other', x: 10, y: 20 + ports[1].top }
		]);
	});

	it('centres a MULTI slot on its two-unit block', () => {
		const [first] = inputAnchors([node({ slots: ['list'], multi: new Set(['list']) })], key);
		expect(first.y, 'a 2-unit slot’s centre is a unit below a 1-unit slot’s').toBe(
			20 + NODE.border + NODE.header + NODE.unit
		);
	});

	it('skips a node with no inputs, and keys every anchor through the caller', () => {
		expect(inputAnchors([node({ slots: [] })], key)).toEqual([]);
		expect(inputAnchors([node({ slots: ['in'] })], (u, s) => `${u}/${s}`)[0].key).toBe('n1/in');
	});
});
