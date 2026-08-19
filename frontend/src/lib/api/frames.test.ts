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

let subscribeFrames: typeof import('./frames').subscribeFrames;
let dropRate: typeof import('./frames').dropRate;

beforeEach(async () => {
	vi.resetModules();
	vi.useFakeTimers();
	MockWorker.instances = [];
	vi.stubGlobal('Worker', MockWorker as unknown as typeof Worker);
	vi.stubGlobal('URL', URL);
	({ subscribeFrames, dropRate } = await import('./frames'));
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
		const offA = subscribeFrames('osc-a', 'out', (f) => gotA.push(f));
		const offB = subscribeFrames('osc-b', 'out', (f) => gotB.push(f));
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

describe('subscribeFrames late join', () => {
	it('replays the slot’s current frame to a late-joining consumer, immediately and once', async () => {
		// The bridge only sends when something changed (an emit, a joiner IT can see, a spec
		// change) — but a consumer joining a stream that is already open in THIS page is
		// invisible to it: no worker traffic happens at all. The frame it needs is already
		// cached on the slot, so the join replays it; without that, a metadata panel joining
		// a slot viewer's stream stares at its empty state until the producer's next emit —
		// ~10 s for a sparse producer, forever for a stopped one.
		const gotA: DataFrame[] = [];
		const offA = subscribeFrames('osc', 'out', (f) => gotA.push(f));
		const frame1 = { shape: [1] } as unknown as DataFrame;
		MockWorker.instances[0].emit({ node: 'osc', slot: 'out', frame: frame1 });
		await vi.advanceTimersByTimeAsync(40); // past the paint scheduler
		expect(gotA, 'the first consumer painted the frame').toEqual([frame1]);

		// The late joiner: no new worker frame, no timers — the cached frame arrives at once.
		const gotB: DataFrame[] = [];
		const offB = subscribeFrames('osc', 'out', (f) => gotB.push(f));
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
		const off = subscribeFrames('osc', 'out', (f) => got.push(f));
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
		const offA = subscribeFrames('osc-a', 'out', () => {});
		const offB = subscribeFrames('osc-b', 'out', () => {});
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

	it('reports null for a stream nobody is watching, once its linger has run out', async () => {
		// Absent is not zero: `0/s` for a stream that is not running asserts something false.
		expect(dropRate('osc-a', 'out')).toBeNull();
		const off = subscribeFrames('osc-a', 'out', () => {});
		expect(dropRate('osc-a', 'out')).toBe(0);
		off();
		// The stream OUTLIVES its last consumer by the linger window — a detach is not yet a
		// departure, because a re-render detaches and re-attaches the same viewer within a tick.
		expect(dropRate('osc-a', 'out'), 'still there while the linger stands').toBe(0);
		await vi.advanceTimersByTimeAsync(500);
		expect(dropRate('osc-a', 'out'), 'and null once nobody came back for it').toBeNull();
	});

	it('a consumer returning inside the linger window keeps the stream it left', async () => {
		// The detach/re-attach a re-render performs. Tearing down on that transient zero closed
		// the socket and dropped the cached frame under every OTHER viewer of the same slot.
		const off = subscribeFrames('osc-a', 'out', () => {});
		const w = MockWorker.instances[0];
		const subs = (): number =>
			w.posted.filter((m) => (m as { op: string }).op === 'sub').length;
		const unsubs = (): number =>
			w.posted.filter((m) => (m as { op: string }).op === 'unsub').length;
		expect(subs()).toBe(1);

		off();
		const back = subscribeFrames('osc-a', 'out', () => {});
		await vi.advanceTimersByTimeAsync(500);
		expect(unsubs(), 'the socket was never given up').toBe(0);
		expect(subs(), 'and never re-opened, because it never closed').toBe(1);
		expect(dropRate('osc-a', 'out')).toBe(0);
		back();
	});
});
