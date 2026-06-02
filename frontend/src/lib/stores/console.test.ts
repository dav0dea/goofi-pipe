import { describe, it, expect } from 'vitest';
import { ConsoleStore, type LogStream } from './console.svelte';

function rec(node: string, stream: LogStream, text: string, seq: number, ts = 0) {
	return { node, stream, text, seq, ts };
}

/** Read an unfiltered view into a plain array. */
function all(s: ConsoleStore) {
	const v = s.acquireView(null, true, true);
	const out = [];
	for (let i = 0; i < v.total(); i++) out.push(v.get(i));
	s.releaseView(v.sig);
	return out;
}

describe('ConsoleStore', () => {
	it('coalesces consecutive identical lines into one ×N entry', () => {
		const s = new ConsoleStore();
		s.ingest(rec('n', 'stdout', 'tick', 0));
		s.ingest(rec('n', 'stdout', 'tick', 1));
		s.ingest(rec('n', 'stdout', 'tick', 2));
		const entries = all(s);
		expect(entries.length).toBe(1);
		expect(entries[0].count).toBe(3);
	});

	it('does not coalesce when a different line interleaves', () => {
		const s = new ConsoleStore();
		s.ingest(rec('a', 'stdout', 'x', 0));
		s.ingest(rec('b', 'stdout', 'x', 0)); // different node breaks the run
		s.ingest(rec('a', 'stdout', 'x', 1));
		expect(all(s).length).toBe(3);
	});

	it('dedups replay by per-node seq (≤ last seen is dropped)', () => {
		const s = new ConsoleStore();
		s.ingest(rec('n', 'stdout', 'a', 5));
		s.ingest(rec('n', 'stdout', 'b', 3)); // stale replay → dropped
		s.ingest(rec('n', 'stdout', 'c', 5)); // equal to last → dropped
		s.ingest(rec('n', 'stdout', 'd', 6)); // fresh → kept
		expect(all(s).map((e) => e.text)).toEqual(['a', 'd']);
	});

	it('seq dedup is per node, not global', () => {
		const s = new ConsoleStore();
		s.ingest(rec('a', 'stdout', 'a0', 0));
		s.ingest(rec('b', 'stdout', 'b0', 0)); // node b's own seq space
		expect(all(s).length).toBe(2);
	});

	it('forgetNodeDedup lets a re-added node restart its sequence', () => {
		const s = new ConsoleStore();
		s.ingest(rec('n', 'stdout', 'old', 9));
		s.forgetNodeDedup('n');
		s.ingest(rec('n', 'stdout', 'new', 0)); // would be dropped without forget
		expect(all(s).map((e) => e.text)).toEqual(['old', 'new']);
	});

	it('filters by node', () => {
		const s = new ConsoleStore();
		s.ingest(rec('a', 'stdout', 'a1', 0));
		s.ingest(rec('b', 'stdout', 'b1', 0));
		s.ingest(rec('a', 'stderr', 'a2', 1));
		const v = s.acquireView('a', true, true);
		const got = [];
		for (let i = 0; i < v.total(); i++) got.push(v.get(i).text);
		expect(got).toEqual(['a1', 'a2']);
	});

	it('filters by stream', () => {
		const s = new ConsoleStore();
		s.ingest(rec('a', 'stdout', 'out', 0));
		s.ingest(rec('a', 'stderr', 'err', 1));
		const vOut = s.acquireView(null, true, false);
		const vErr = s.acquireView(null, false, true);
		const read = (v: ReturnType<ConsoleStore['acquireView']>) => {
			const o = [];
			for (let i = 0; i < v.total(); i++) o.push(v.get(i).text);
			return o;
		};
		expect(read(vOut)).toEqual(['out']);
		expect(read(vErr)).toEqual(['err']);
	});

	it('keeps a filtered view in sync as new matching lines arrive', () => {
		const s = new ConsoleStore();
		const v = s.acquireView('a', true, true);
		s.ingest(rec('a', 'stdout', 'a1', 0));
		s.ingest(rec('b', 'stdout', 'b1', 0)); // not in view
		s.ingest(rec('a', 'stdout', 'a2', 1));
		expect(v.total()).toBe(2);
		expect(v.get(0).text).toBe('a1');
		expect(v.get(1).text).toBe('a2');
	});

	it('mirrors errors as stderr, bypassing seq dedup, and coalesces repeats', () => {
		const s = new ConsoleStore();
		s.ingest(rec('n', 'stdout', 'hi', 100)); // advance the seq cursor high
		s.ingestError('n', 'Traceback...\nBoom', 0); // no seq — still ingested
		s.ingestError('n', 'Traceback...\nBoom', 0); // identical → ×2
		const entries = all(s);
		expect(entries.length).toBe(2);
		expect(entries[1].stream).toBe('stderr');
		expect(entries[1].count).toBe(2);
	});

	it('keeps a coalesced repeat as one entry (uid identity preserved)', () => {
		const s = new ConsoleStore();
		s.ingest(rec('n', 'stdout', 'x', 0));
		s.ingest(rec('n', 'stdout', 'x', 1)); // coalesce → no new uid
		s.ingest(rec('n', 'stdout', 'y', 2)); // new entry
		const v = s.acquireView(null, true, true);
		expect(v.total()).toBe(2);
		expect(v.get(0).uid).not.toBe(v.get(1).uid);
	});

	it('groups a filtered node consecutively even when another node interleaves', () => {
		const s = new ConsoleStore();
		const va = s.acquireView('a', true, true); // filter active before ingest
		s.ingest(rec('a', 'stderr', 'boom', 0));
		s.ingest(rec('b', 'stdout', 'noise', 0)); // interleaves in the global stream
		s.ingest(rec('a', 'stderr', 'boom', 1)); // same as a's previous line
		// Unfiltered: three separate entries (a's run was interrupted by b).
		const all = s.acquireView(null, true, true);
		expect(all.total()).toBe(3);
		// Filtered to a: the two 'boom' are consecutive for a → one ×2 entry.
		expect(va.total()).toBe(1);
		expect(va.get(0).count).toBe(2);
	});

	it('seeds a filtered view from history, re-coalescing for the filter', () => {
		const s = new ConsoleStore();
		s.ingest(rec('a', 'stderr', 'boom', 0));
		s.ingest(rec('b', 'stdout', 'noise', 0));
		s.ingest(rec('a', 'stderr', 'boom', 1));
		// Acquire AFTER ingest → built from the unfiltered view's history.
		const va = s.acquireView('a', true, true);
		expect(va.total()).toBe(1);
		expect(va.get(0).count).toBe(2);
	});

	it('clear() empties the buffer', () => {
		const s = new ConsoleStore();
		s.ingest(rec('n', 'stdout', 'x', 0));
		s.clear();
		expect(all(s).length).toBe(0);
	});

	it('caps at 100k entries, dropping the oldest first', () => {
		const s = new ConsoleStore();
		const N = 100_000;
		for (let i = 0; i < N + 5; i++) s.ingest(rec('n', 'stdout', `line-${i}`, i));
		const v = s.acquireView(null, true, true);
		expect(v.total()).toBe(N);
		// The first 5 were evicted; oldest surviving is line-5.
		expect(v.get(0).text).toBe('line-5');
		expect(v.get(v.total() - 1).text).toBe(`line-${N + 4}`);
	});
});
