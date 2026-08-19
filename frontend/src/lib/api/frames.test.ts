import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DataFrame } from '$lib/codec/decode';

/** Minimal Worker stand-in (the data.test.ts idiom): records postMessage, emits inbound. */
class MockWorker {
	static instances: MockWorker[] = [];
	posted: unknown[] = [];
	private listeners: ((e: MessageEvent) => void)[] = [];
	constructor() {
		MockWorker.instances.push(this);
	}
	postMessage(m: unknown): void {
		this.posted.push(m);
	}
	addEventListener(_type: string, cb: (e: MessageEvent) => void): void {
		this.listeners.push(cb);
	}
	emit(data: unknown): void {
		for (const cb of this.listeners) cb({ data } as MessageEvent);
	}
	terminate(): void {}
}

let bindViewer: typeof import('./frames').bindViewer;
let dropRate: typeof import('./frames').dropRate;

/** Let the reconcile microtask run. */
const settle = (): Promise<void> => Promise.resolve();

let seq = 0;
/** Bind a viewer that constrains nothing, with a token of its own. */
const bind = (
	node: string,
	slot: string,
	cb: (f: DataFrame) => void = () => {}
): (() => void) => bindViewer(node, slot, `v${++seq}`, null, cb);

const line = (max: number) => ({
	dtype: 'array' as const,
	ndim: [['le', 3]] as [import('$lib/viewers/capacity').DimCmp, number][],
	dims: [],
	reduce: [{ dim: -1, max, method: 'envelope' as const }]
});

const opsOf = (w: MockWorker, op: string): unknown[] =>
	w.posted.filter((m) => (m as { op: string }).op === op);

beforeEach(async () => {
	vi.resetModules();
	vi.useFakeTimers();
	MockWorker.instances = [];
	vi.stubGlobal('Worker', MockWorker as unknown as typeof Worker);
	vi.stubGlobal('URL', URL);
	seq = 0;
	({ bindViewer, dropRate } = await import('./frames'));
});

afterEach(() => {
	vi.useRealTimers();
	vi.unstubAllGlobals();
});

describe('paint-rate accounting', () => {
	it('records one paint per flush, however many streams painted in it', async () => {
		// The HUD's "fps" is the PAINT rate — flushes/s, which the cap (paintCap.ts) bounds at 30
		// app-wide. It used to be bumped inside the per-slot loop, so it read the SUM of the open
		// streams' delivery rates: ~30 x N, climbing by 30 with every node the user added while the
		// cap it was being used to instrument never moved. TWO streams is the smallest fixture that
		// can tell the two arithmetics apart — at N=1 the sum and the paint rate are the same number,
		// which is exactly why the single-node `viewer-fps-cap.spec.ts` stayed green through it.
		const { perfStats } = await import('./perfStats.svelte');
		const delivered = vi.spyOn(perfStats(), 'delivered');
		const gotA: DataFrame[] = [];
		const gotB: DataFrame[] = [];
		const offA = bind('osc-a', 'out', (f) => gotA.push(f));
		const offB = bind('osc-b', 'out', (f) => gotB.push(f));
		await settle(); // the reconcile opens the streams
		const w = MockWorker.instances[0];
		w.emit({ node: 'osc-a', slot: 'out', frame: { shape: [1] } as unknown as DataFrame });
		w.emit({ node: 'osc-b', slot: 'out', frame: { shape: [2] } as unknown as DataFrame });
		await vi.advanceTimersByTimeAsync(40); // past the paint scheduler → ONE flush
		// The fixture is only honest if both streams really painted in that one flush.
		expect(gotA.length, 'stream A painted').toBe(1);
		expect(gotB.length, 'stream B painted').toBe(1);
		expect(delivered, 'one flush is one paint, not one per slot').toHaveBeenCalledTimes(1);

		// …and it is a rate, not a latch: the next flush counts too.
		w.emit({ node: 'osc-a', slot: 'out', frame: { shape: [3] } as unknown as DataFrame });
		w.emit({ node: 'osc-b', slot: 'out', frame: { shape: [4] } as unknown as DataFrame });
		await vi.advanceTimersByTimeAsync(40);
		expect(delivered, 'a second flush is a second paint').toHaveBeenCalledTimes(2);
		offA();
		offB();
	});
});

describe('a joining viewer', () => {
	it('replays the slot’s current frame to a late-joining consumer, immediately and once', async () => {
		// The bridge only sends when something changed (an emit, a joiner IT can see, a spec
		// change) — but a consumer joining a stream that is already open in THIS page is
		// invisible to it: no worker traffic happens at all. The frame it needs is already
		// cached on the slot, so the join replays it; without that, a metadata panel joining
		// a slot viewer's stream stares at its empty state until the producer's next emit —
		// ~10 s for a sparse producer, forever for a stopped one.
		const gotA: DataFrame[] = [];
		const offA = bind('osc', 'out', (f) => gotA.push(f));
		await settle();
		const frame1 = { shape: [1] } as unknown as DataFrame;
		MockWorker.instances[0].emit({ node: 'osc', slot: 'out', frame: frame1 });
		await vi.advanceTimersByTimeAsync(40); // past the paint scheduler
		expect(gotA, 'the first consumer painted the frame').toEqual([frame1]);

		// The late joiner: no new worker frame, no timers — the cached frame arrives at once.
		const gotB: DataFrame[] = [];
		const offB = bind('osc', 'out', (f) => gotB.push(f));
		expect(gotB, 'a late joiner is served the cached current frame').toEqual([frame1]);
		expect(gotA, 'the replay reaches only the joiner, not the settled consumers').toEqual([
			frame1
		]);

		// And it is a replay, not a re-delivery loop: nothing further arrives unprompted.
		await vi.advanceTimersByTimeAsync(200);
		expect(gotB).toEqual([frame1]);
		offA();
		offB();
	});

	it('a joiner of a slot with no frame yet is not called', () => {
		const got: DataFrame[] = [];
		const off = bind('osc', 'out', (f) => got.push(f));
		expect(got).toEqual([]);
		off();
	});
});

describe('per-stream drop accounting', () => {
	it('attributes a coalesced frame to the stream that dropped it, not to the app', async () => {
		// A "drop" is latest-wins coalescing: a frame overwritten before it painted. It was summed
		// app-wide beside an fps counter that is emphatically NOT a sum, so the pair could not be
		// read together. TWO streams is the smallest fixture that can tell "per stream" from
		// "app-wide" — with one stream both arithmetics give the same number.
		const offA = bind('osc-a', 'out');
		const offB = bind('osc-b', 'out');
		await settle();
		const w = MockWorker.instances[0];
		// Two frames on A before any flush → the first is coalesced away. B gets one, so it drops none.
		w.emit({ node: 'osc-a', slot: 'out', frame: { shape: [1] } as unknown as DataFrame });
		w.emit({ node: 'osc-a', slot: 'out', frame: { shape: [2] } as unknown as DataFrame });
		w.emit({ node: 'osc-b', slot: 'out', frame: { shape: [3] } as unknown as DataFrame });
		await vi.advanceTimersByTimeAsync(600); // past the meter window

		expect(dropRate('osc-a', 'out')).toBeGreaterThan(0);
		expect(dropRate('osc-b', 'out'), 'the quiet stream is not charged for its neighbour').toBe(0);
		offA();
		offB();
	});

	it('reports null for a stream nobody is watching', async () => {
		// Absent is not zero: `0/s` for a stream that is not running asserts something false.
		expect(dropRate('osc-a', 'out')).toBeNull();
		const off = bind('osc-a', 'out');
		await settle();
		expect(dropRate('osc-a', 'out')).toBe(0);
		off();
		await settle();
		expect(dropRate('osc-a', 'out'), 'and null again once the last viewer leaves').toBeNull();
	});
});

/**
 * The registry is the ONLY thing that decides what the backend is asked for, and it is read once
 * the tick has settled rather than per mutation. That is what makes a viewer's comings and goings
 * invisible to the backend unless they actually change what it should serve.
 */
describe('what the registry tells the backend', () => {
	it('opens once for a slot, whatever the viewer count, and closes when the last one goes', async () => {
		const offA = bind('osc', 'out');
		const offB = bind('osc', 'out');
		await settle();
		const w = MockWorker.instances[0];
		expect(opsOf(w, 'sub'), 'two viewers, one stream').toEqual([
			{ op: 'sub', node: 'osc', slot: 'out' }
		]);

		offA();
		await settle();
		expect(opsOf(w, 'unsub'), 'one viewer left, and it is still watched').toEqual([]);
		offB();
		await settle();
		expect(opsOf(w, 'unsub')).toEqual([{ op: 'unsub', node: 'osc', slot: 'out' }]);
	});

	it('says NOTHING when a viewer leaves a slot another viewer is still on', async () => {
		// The failure this replaces: the node's own viewer closing tore down the stream a panel
		// bound to the same slot was drawing from. Nothing about what the backend should serve
		// changed, so nothing should reach it — not an unsub, not a re-sub, not a spec.
		const offA = bindViewer('osc', 'out', 'a', line(256), () => {});
		bindViewer('osc', 'out', 'b', line(256), () => {});
		await settle();
		const w = MockWorker.instances[0];
		const before = w.posted.length;

		offA();
		await settle();
		expect(w.posted.length, `the backend was told: ${JSON.stringify(w.posted.slice(before))}`).toBe(
			before
		);
	});

	it('says nothing when a viewer detaches and re-attaches inside one tick', async () => {
		// What a re-render does. The registry ends the tick exactly as it started, so the reconcile
		// finds nothing to say — no socket churn, and no cached frame thrown away.
		const off = bindViewer('osc', 'out', 'a', line(256), () => {});
		await settle();
		const w = MockWorker.instances[0];
		const before = w.posted.length;

		off();
		bindViewer('osc', 'out', 'a', line(256), () => {});
		await settle();
		expect(w.posted.length, 'a detach and re-attach is not an event').toBe(before);
	});

	it('sends the specs only when the list actually changes', async () => {
		bindViewer('osc', 'out', 'a', line(150), () => {});
		await settle();
		const w = MockWorker.instances[0];
		expect(opsOf(w, 'spec')).toEqual([
			{ op: 'spec', node: 'osc', slot: 'out', specs: [line(150)] }
		]);

		// Re-binding the same viewer with the same need — a re-render — says nothing.
		bindViewer('osc', 'out', 'a', line(150), () => {});
		await settle();
		expect(opsOf(w, 'spec'), 'an unchanged need is not renegotiated').toHaveLength(1);

		// A resize is a real change, and it is sent.
		bindViewer('osc', 'out', 'a', line(320), () => {});
		await settle();
		expect(opsOf(w, 'spec')).toHaveLength(2);
		expect((opsOf(w, 'spec').at(-1) as { specs: unknown[] }).specs).toEqual([line(320)]);
	});

	it('collects every viewer’s need, and lets a reader contribute none', async () => {
		// Sent verbatim as a LIST: the bridge folds them, because only it has the real frame to
		// drop the ones a shape rules out. A null-spec viewer (the metadata panel) reads the same
		// stream without narrowing it to a budget it never asked for.
		bindViewer('osc', 'out', 'wide', line(2000), () => {});
		bindViewer('osc', 'out', 'narrow', line(150), () => {});
		bindViewer('osc', 'out', 'reader', null, () => {});
		await settle();
		const w = MockWorker.instances[0];
		expect((opsOf(w, 'spec').at(-1) as { specs: unknown[] }).specs).toEqual([
			line(2000),
			line(150)
		]);
	});
});
