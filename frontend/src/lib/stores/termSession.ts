/**
 * The terminal store — one live `Terminal` per harness instance, kept OUTSIDE the panel.
 *
 * This is the whole of the re-attach design (user, 2026-08-10). The manager emulates no terminal
 * and keeps no grid: it is a pipe. The screen and the scrollback are the xterm `Terminal` object's
 * own, so keeping that object alive here — keyed by `instance_id`, not held by the component —
 * means closing a panel and opening another loses nothing, and re-attaching is `term.open(newEl)`
 * rather than a replay. Nothing survives a page reload or reaches a second tab; that is the
 * explicit trade the no-emulator decision made, and why there is no ring buffer here either.
 *
 * The socket therefore outlives the panel: an unmounted view keeps receiving, so what the agent
 * wrote while nobody was looking is in the buffer when someone looks again. What it gives up on
 * unmount is its say in the SIZE — see `retract`.
 *
 * `TerminalLike` is xterm's surface narrowed to the calls this drives, so this module is a
 * plain unit-tested one: xterm itself needs a DOM, and the component that owns the import is
 * verified by typecheck + e2e.
 */
export interface TerminalLike {
	open(el: HTMLElement): void;
	/** What `open` built — undefined until it has. xterm exposes this, and `attach` needs it. */
	readonly element: HTMLElement | undefined;
	write(data: Uint8Array | string): void;
	resize(cols: number, rows: number): void;
	onData(cb: (data: string) => void): { dispose(): void };
	/** The fit addon's measurement of the container, or undefined before there is one to measure.
	 * Composed onto the terminal at construction, because the addon has to be loaded into the
	 * terminal that OUTLIVES the panel — a component that kept its own would have none to ask
	 * after a remount. */
	proposeDimensions(): { cols: number; rows: number } | undefined;
	dispose(): void;
}

/** One instance's terminal and the `/term` socket feeding it. */
export class TermSession {
	private ws: WebSocket | null = null;
	/** Armed by every attach: the next proposal goes out as cols−1 then cols, which is what makes
	 * a full-screen TUI redraw itself onto a screen it has already laid out once. */
	private nudge = false;
	/** The last proposal this view put on the wire, so a settling `ResizeObserver` does not chatter
	 * the same numbers at the manager. */
	private said = '';

	constructor(
		readonly id: string,
		readonly term: TerminalLike
	) {
		term.onData((d) => this.send(new TextEncoder().encode(d)));
		this.open();
	}

	private open(): void {
		const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
		const ws = new WebSocket(`${proto}//${location.host}/term/${encodeURIComponent(this.id)}`);
		ws.binaryType = 'arraybuffer';
		ws.addEventListener('message', (e: MessageEvent) => this.receive(e.data));
		this.ws = ws;
		this.said = '';
	}

	/** Binary frames are PTY bytes; text frames are control. */
	private receive(data: unknown): void {
		if (data instanceof ArrayBuffer) {
			this.term.write(new Uint8Array(data));
			return;
		}
		const msg = JSON.parse(String(data)) as { op?: string; cols?: number; rows?: number };
		// THE INVARIANT. An authoritative size sets the terminal DIRECTLY and never reaches the fit
		// addon: one PTY window is shared by every view, so a view that answered an inbound size by
		// re-measuring and proposing would put two views in a loop proposing each other's sizes
		// forever. Only a container resize proposes (see `propose`); everything else obeys.
		if (msg.op === 'size' && msg.cols && msg.rows) this.term.resize(msg.cols, msg.rows);
	}

	private send(bytes: Uint8Array | string): void {
		if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(bytes);
	}

	private resize(cols: number, rows: number): void {
		this.said = `${cols}x${rows}`;
		this.send(JSON.stringify({ op: 'resize', cols, rows }));
	}

	/** Draw this session's terminal into `el` — on first mount, and on every remount after.
	 *
	 * A remount MOVES the element the first open built rather than opening again: xterm's `open()`
	 * does nothing at all once the terminal has an element (from 5.3 it only re-points the window),
	 * so calling it a second time leaves the new panel with an empty box — the very panel that is
	 * supposed to have lost nothing. Moving the element is also what carries the screen and the
	 * scrollback across, since they belong to the terminal that built it. */
	attach(el: HTMLElement): void {
		if (!this.ws) this.open();
		const drawn = this.term.element;
		if (drawn) el.appendChild(drawn);
		else this.term.open(el);
		this.nudge = true;
	}

	/** The ONE writer path: what this view's container measured. */
	propose(cols: number, rows: number): void {
		if (this.nudge) {
			this.nudge = false;
			this.resize(Math.max(1, cols - 1), rows);
		} else if (this.said === `${cols}x${rows}`) {
			return;
		}
		this.resize(cols, rows);
	}

	/** Measure and propose — what a container `ResizeObserver` fires, and the ONLY thing that does.
	 * Everything else in this module obeys the size it is given. */
	refit(): void {
		const d = this.term.proposeDimensions();
		if (d) this.propose(d.cols, d.rows);
	}

	/** This view has nothing on screen to speak for (its panel unmounted), so it stops speaking —
	 * the manager hands the size back to whichever view still does. The socket stays: that is what
	 * keeps the transcript arriving for the next time this instance is shown. */
	retract(): void {
		this.resize(0, 0);
	}

	/** The user's explicit Detach: drop this VIEW. The socket closes, which is exactly the event
	 * the manager re-arbitrates the size on; the terminal and its scrollback stay, so re-attaching
	 * from the switcher picks the transcript back up. */
	detach(): void {
		this.ws?.close();
		this.ws = null;
	}

	dispose(): void {
		this.detach();
		this.term.dispose();
	}
}

const live = new Map<string, TermSession>();

/** This instance's session, minting it (and its terminal) on first ask. */
export function termSession(id: string, make: () => TerminalLike): TermSession {
	let s = live.get(id);
	if (!s) {
		s = new TermSession(id, make());
		live.set(id, s);
	}
	return s;
}

/** Drop an instance's session for good — what an instance leaving the roster means. */
export function endTermSession(id: string): void {
	live.get(id)?.dispose();
	live.delete(id);
}

/** The instances a terminal is being kept for, so the roster can end the ones that are gone. */
export function liveTermSessions(): string[] {
	return [...live.keys()];
}
