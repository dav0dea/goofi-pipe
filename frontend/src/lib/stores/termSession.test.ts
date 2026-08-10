import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { TerminalLike } from './termSession';

/** A stand-in for xterm's own element: what `open()` BUILDS and every later attach moves. */
class FakeEl {
	kids: unknown[] = [];
	constructor(readonly name: string) {}
	appendChild(c: unknown): void {
		this.kids.push(c);
	}
}

/** A stand-in for xterm's `Terminal` — the calls the session drives, recorded. `open` builds an
 * element and, like the real one from 5.3 on, does nothing at all once it has. */
class FakeTerm implements TerminalLike {
	element: HTMLElement | undefined;
	written: string[] = [];
	opened: unknown[] = [];
	size: [number, number] | null = null;
	disposed = false;
	fits: { cols: number; rows: number } | undefined;
	private typed: ((d: string) => void) | null = null;
	open(el: HTMLElement): void {
		this.opened.push(el);
		this.element ??= new FakeEl('xterm') as unknown as HTMLElement;
	}
	write(d: Uint8Array | string): void {
		this.written.push(typeof d === 'string' ? d : new TextDecoder().decode(d));
	}
	resize(cols: number, rows: number): void {
		this.size = [cols, rows];
	}
	onData(cb: (d: string) => void): { dispose(): void } {
		this.typed = cb;
		return { dispose() {} };
	}
	proposeDimensions(): { cols: number; rows: number } | undefined {
		return this.fits;
	}
	dispose(): void {
		this.disposed = true;
	}
	/** The user pressing a key. */
	type(s: string): void {
		this.typed?.(s);
	}
}

class FakeSocket {
	static all: FakeSocket[] = [];
	static OPEN = 1;
	binaryType = '';
	readyState = 1;
	sent: unknown[] = [];
	closed = false;
	private listeners: Record<string, ((e: unknown) => void)[]> = {};
	constructor(readonly url: string) {
		FakeSocket.all.push(this);
	}
	addEventListener(t: string, cb: (e: unknown) => void): void {
		(this.listeners[t] ??= []).push(cb);
	}
	send(d: unknown): void {
		this.sent.push(d);
	}
	close(): void {
		this.closed = true;
		this.readyState = 3;
	}
	/** A frame from the manager. */
	deliver(data: unknown): void {
		for (const cb of this.listeners.message ?? []) cb({ data });
	}
	/** Everything this socket sent, JSON control frames parsed. */
	control(): Record<string, unknown>[] {
		return this.sent
			.filter((s): s is string => typeof s === 'string')
			.map((s) => JSON.parse(s) as Record<string, unknown>);
	}
}

let mod: typeof import('./termSession');
let term: FakeTerm;

beforeEach(async () => {
	vi.resetModules();
	FakeSocket.all = [];
	vi.stubGlobal('WebSocket', FakeSocket as unknown as typeof WebSocket);
	vi.stubGlobal('location', { protocol: 'http:', host: 'here:8000' });
	mod = await import('./termSession');
	term = new FakeTerm();
});

afterEach(() => {
	vi.unstubAllGlobals();
});

const el = (name: string): HTMLElement => new FakeEl(name) as unknown as HTMLElement;
const sock = (): FakeSocket => FakeSocket.all[FakeSocket.all.length - 1];

describe('the terminal store', () => {
	it('keeps ONE Terminal per instance, so a panel can close and reopen without losing it', () => {
		const first = mod.termSession('abc', () => term);
		first.attach(el('panel-1'));
		sock().deliver(new TextEncoder().encode('hello agent').buffer);

		// The panel unmounts: it stops speaking for the size, but the socket and the buffer stay.
		first.retract();
		expect(sock().closed, 'an unmount does not drop the stream').toBe(false);
		sock().deliver(new TextEncoder().encode(' — still running').buffer);

		// …and a NEW panel asking for the same instance is handed the same terminal, redrawn into
		// its element. Nothing was replayed: the scrollback is the Terminal object's own.
		const again = mod.termSession('abc', () => new FakeTerm());
		expect(again.term, 'a second ask mints a second terminal').toBe(term);
		const second = el('panel-2');
		again.attach(second);
		// xterm's `open()` is a no-op once the terminal HAS an element (5.3 onwards it only
		// re-points the window), so a remount has to MOVE what the first open built. Opening again
		// leaves the new panel with an empty box — which is precisely the panel that is supposed to
		// have lost nothing.
		expect((second as unknown as FakeEl).kids, 'the terminal was moved into the new panel')
			.toEqual([term.element]);
		expect(term.opened, 'and never opened twice').toHaveLength(1);
		expect(term.written.join('')).toBe('hello agent — still running');
		expect(FakeSocket.all).toHaveLength(1);
	});

	it('carries bytes both ways', () => {
		const s = mod.termSession('abc', () => term);
		expect(sock().url).toBe('ws://here:8000/term/abc');
		expect(sock().binaryType).toBe('arraybuffer');
		s.attach(el('p'));
		term.type('ls\r');
		expect(sock().sent.at(-1)).toEqual(new TextEncoder().encode('ls\r'));
	});

	it('sizes the terminal from the manager ALONE, and never answers a size with a size', () => {
		const s = mod.termSession('abc', () => term);
		s.attach(el('p'));
		s.propose(80, 24);
		const asked = sock().control().length;

		sock().deliver(JSON.stringify({ op: 'size', cols: 120, rows: 40 }));
		expect(term.size, 'an authoritative size sets the terminal directly').toEqual([120, 40]);
		expect(sock().control()).toHaveLength(asked);

		// The invariant, stated as the loop it prevents: whatever arrives, this view proposes only
		// what its own container measured. A size that fed back into the fit addon would have two
		// views answering each other's frames forever.
		sock().deliver(JSON.stringify({ op: 'size', cols: 200, rows: 60 }));
		expect(sock().control()).toHaveLength(asked);
	});

	it('nudges the size on attach, so a full-screen TUI repaints itself', () => {
		const s = mod.termSession('abc', () => term);
		s.attach(el('p'));
		s.propose(100, 30);
		expect(sock().control()).toEqual([
			{ op: 'resize', cols: 99, rows: 30 },
			{ op: 'resize', cols: 100, rows: 30 }
		]);

		// Only on attach: a container that settles at the same size afterwards says nothing more.
		s.propose(100, 30);
		expect(sock().control()).toHaveLength(2);
		s.propose(90, 30);
		expect(sock().control().at(-1)).toEqual({ op: 'resize', cols: 90, rows: 30 });
	});

	it('proposes what the fit addon measured, and says nothing when it measured nothing', () => {
		const s = mod.termSession('abc', () => term);
		s.attach(el('p'));
		s.refit();
		expect(sock().control(), 'a terminal with no laid-out box proposes nothing').toEqual([]);
		term.fits = { cols: 64, rows: 16 };
		s.refit();
		expect(sock().control().at(-1)).toEqual({ op: 'resize', cols: 64, rows: 16 });
	});

	it('retracts with a zero size, and proposes again on the next attach', () => {
		const s = mod.termSession('abc', () => term);
		s.attach(el('p'));
		s.propose(100, 30);
		s.retract();
		expect(sock().control().at(-1)).toEqual({ op: 'resize', cols: 0, rows: 0 });
		s.attach(el('p2'));
		s.propose(100, 30);
		expect(sock().control().slice(-2)).toEqual([
			{ op: 'resize', cols: 99, rows: 30 },
			{ op: 'resize', cols: 100, rows: 30 }
		]);
	});

	it('detaches by dropping the socket and keeping the terminal; ending drops both', () => {
		const s = mod.termSession('abc', () => term);
		s.attach(el('p'));
		s.detach();
		expect(sock().closed, 'the manager sees the detach and re-arbitrates the size').toBe(true);
		expect(term.disposed, 'the transcript survives a detach').toBe(false);
		expect(mod.liveTermSessions()).toEqual(['abc']);

		// Re-attaching from the switcher opens a fresh socket onto the SAME terminal.
		s.attach(el('p'));
		expect(FakeSocket.all).toHaveLength(2);
		expect(sock().closed).toBe(false);

		mod.endTermSession('abc');
		expect(term.disposed).toBe(true);
		expect(sock().closed).toBe(true);
		expect(mod.liveTermSessions()).toEqual([]);
	});
});
