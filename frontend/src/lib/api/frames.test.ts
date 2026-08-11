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

beforeEach(async () => {
	vi.resetModules();
	vi.useFakeTimers();
	MockWorker.instances = [];
	vi.stubGlobal('Worker', MockWorker as unknown as typeof Worker);
	vi.stubGlobal('URL', URL);
	({ subscribeFrames } = await import('./frames'));
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
