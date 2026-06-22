import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

/** Minimal Worker stand-in: records postMessage and lets a test emit messages. */
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

let subscribeData: typeof import('./data').subscribeData;

beforeEach(async () => {
	vi.resetModules();
	MockWorker.instances = [];
	vi.stubGlobal('Worker', MockWorker as unknown as typeof Worker);
	vi.stubGlobal('URL', URL); // new Worker(new URL(...)) — keep real URL
	({ subscribeData } = await import('./data'));
});

afterEach(() => vi.unstubAllGlobals());

describe('subscribeData (kind-aware)', () => {
	it('subscribes per (node, slot, kind) and routes frames by kind', () => {
		const got = { image: 0, line: 0 };
		const offImg = subscribeData('osc', 'out', 'image', () => got.image++);
		const offLine = subscribeData('osc', 'out', 'line', () => got.line++);

		const w = MockWorker.instances[0];
		expect(w.posted).toEqual([
			{ op: 'sub', node: 'osc', slot: 'out', kind: 'image' },
			{ op: 'sub', node: 'osc', slot: 'out', kind: 'line' }
		]);

		// a frame for the 'line' kind reaches only the line consumer
		w.emit({ node: 'osc', slot: 'out', kind: 'line', frame: {} });
		expect(got).toEqual({ image: 0, line: 1 });

		offLine();
		expect(w.posted.at(-1)).toEqual({ op: 'unsub', node: 'osc', slot: 'out', kind: 'line' });
		offImg();
	});

	it('shares one worker subscription among same-kind consumers', () => {
		const off1 = subscribeData('n', 's', 'line', () => {});
		const off2 = subscribeData('n', 's', 'line', () => {});
		const w = MockWorker.instances[0];
		// one 'sub' for both consumers
		expect(w.posted.filter((m) => (m as { op: string }).op === 'sub')).toHaveLength(1);
		off1();
		// still one consumer → no unsub yet
		expect(w.posted.some((m) => (m as { op: string }).op === 'unsub')).toBe(false);
		off2();
		expect(w.posted.at(-1)).toEqual({ op: 'unsub', node: 'n', slot: 's', kind: 'line' });
	});
});
