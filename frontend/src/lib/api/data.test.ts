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
let setViewSpec: typeof import('./data').setViewSpec;
let clearViewSpec: typeof import('./data').clearViewSpec;

beforeEach(async () => {
	vi.resetModules();
	MockWorker.instances = [];
	vi.stubGlobal('Worker', MockWorker as unknown as typeof Worker);
	vi.stubGlobal('URL', URL); // new Worker(new URL(...)) — keep real URL
	({ subscribeData, setViewSpec, clearViewSpec } = await import('./data'));
});

afterEach(() => vi.unstubAllGlobals());

const flush = (): Promise<void> => Promise.resolve(); // let queued microtasks run

const line = (max: number) => ({
	dtype: 'array' as const,
	ndim: [['le', 3]] as [import('$lib/viewers/capacity').DimCmp, number][],
	dims: [],
	reduce: [{ dim: -1, max, method: 'envelope' as const }]
});

describe('subscribeData (one stream per node/slot)', () => {
	it('subscribes per (node, slot) and routes frames to every consumer', () => {
		const got = { a: 0, b: 0 };
		const offA = subscribeData('osc', 'out', () => got.a++);
		const offB = subscribeData('osc', 'out', () => got.b++);

		const w = MockWorker.instances[0];
		// One 'sub' for the slot — the second consumer shares it (no per-kind sub).
		expect(w.posted).toEqual([{ op: 'sub', node: 'osc', slot: 'out' }]);

		// A frame for the slot reaches BOTH consumers.
		w.emit({ node: 'osc', slot: 'out', frame: {} });
		expect(got).toEqual({ a: 1, b: 1 });

		offA();
		// Still one consumer → no unsub yet.
		expect(w.posted.some((m) => (m as { op: string }).op === 'unsub')).toBe(false);
		offB();
		expect(w.posted.at(-1)).toEqual({ op: 'unsub', node: 'osc', slot: 'out' });
	});

	it('a different slot gets its own stream', () => {
		subscribeData('osc', 'out', () => {});
		subscribeData('osc', 'other', () => {});
		const w = MockWorker.instances[0];
		expect(w.posted.filter((m) => (m as { op: string }).op === 'sub')).toHaveLength(2);
	});
});

describe('ViewSpec contribution → inband list', () => {
	it('posts the UNION of viewers’ specs as a list (bridge folds, not the client)', async () => {
		subscribeData('n', 's', () => {});
		setViewSpec('n', 's', 'v1', line(150));
		setViewSpec('n', 's', 'v2', line(2000));
		await flush();
		const w = MockWorker.instances[0];
		const specMsg = w.posted.filter((m) => (m as { op: string }).op === 'spec').at(-1) as {
			op: string;
			node: string;
			slot: string;
			specs: unknown[];
		};
		// Both contributions are sent verbatim — no client-side fold.
		expect(specMsg).toEqual({ op: 'spec', node: 'n', slot: 's', specs: [line(150), line(2000)] });
	});

	it('coalesces a clear→re-add within one tick into a single non-empty post', async () => {
		subscribeData('n', 's', () => {});
		setViewSpec('n', 's', 'v1', line(150));
		await flush();
		const w = MockWorker.instances[0];
		const posts = () => w.posted.filter((m) => (m as { op: string }).op === 'spec');
		expect(posts()).toHaveLength(1);

		// A resize fires clear() then set() in the same tick — collapses to ONE post,
		// and never a transient empty list (which would ship a full-resolution frame).
		clearViewSpec('n', 's', 'v1');
		setViewSpec('n', 's', 'v1', line(320));
		await flush();
		expect(posts()).toHaveLength(2);
		expect((posts().at(-1) as { specs: unknown[] }).specs).toEqual([line(320)]);
	});

	it('clearing the last spec posts an empty list (drains to full-resolution)', async () => {
		subscribeData('n', 's', () => {});
		setViewSpec('n', 's', 'v1', line(150));
		await flush();
		clearViewSpec('n', 's', 'v1');
		await flush();
		const w = MockWorker.instances[0];
		const last = w.posted.filter((m) => (m as { op: string }).op === 'spec').at(-1) as {
			specs: unknown[];
		};
		expect(last.specs).toEqual([]);
	});
});
