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
