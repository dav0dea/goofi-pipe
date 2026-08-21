/** One live xterm `Terminal` and `/term` socket per harness instance, both kept OUTSIDE the panel
 * so a close and re-open loses nothing. `TerminalLike` narrows xterm to the calls this drives. */
export interface TerminalLike {
	open(el: HTMLElement): void;
	/** What `open` built — undefined until it has. */
	readonly element: HTMLElement | undefined;
	write(data: Uint8Array | string): void;
	resize(cols: number, rows: number): void;
	onData(cb: (data: string) => void): { dispose(): void };
	/** The fit addon's measurement of the container; composed onto the terminal, which outlives the
	 * panel, so a remount still has an addon to ask. */
	proposeDimensions(): { cols: number; rows: number } | undefined;
	dispose(): void;
}

/** One instance's terminal and the `/term` socket feeding it. */
export class TermSession {
	private ws: WebSocket | null = null;
	/** Armed by every attach: the next proposal goes out as cols−1 then cols, which makes a
	 * full-screen TUI redraw itself onto a screen it has already laid out once. */
	private nudge = false;
	/** The last proposal this view put on the wire, so a settling `ResizeObserver` does not chatter. */
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
		// The panel measures its container INSIDE this handshake, so that first proposal is dropped.
		ws.addEventListener('open', () => this.refit());
		// Nothing else nulls it, so without this `attach` reuses a dead connection for ever.
		ws.addEventListener('close', () => {
			if (this.ws === ws) this.ws = null;
		});
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
		// An inbound size sets the terminal DIRECTLY and never re-measures: one PTY window is shared,
		// so a view that answered by proposing would loop with every other view.
		if (msg.op === 'size' && msg.cols && msg.rows) this.term.resize(msg.cols, msg.rows);
	}

	/** Whether it went out — a socket that is not yet OPEN silently drops what is written to it. */
	private send(bytes: Uint8Array<ArrayBuffer> | string): boolean {
		if (this.ws?.readyState !== WebSocket.OPEN) return false;
		this.ws.send(bytes);
		return true;
	}

	private resize(cols: number, rows: number): boolean {
		const out = this.send(JSON.stringify({ op: 'resize', cols, rows }));
		if (out) this.said = `${cols}x${rows}`;
		return out;
	}

	/** Draw this session's terminal into `el`. A remount MOVES the element the first open built:
	 * xterm's `open()` does nothing at all once the terminal already has one. */
	attach(el: HTMLElement): void {
		if (!this.ws) this.open();
		// A panel switching instances keeps its element, so appending beside would stack two terminals.
		el.replaceChildren();
		const drawn = this.term.element;
		if (drawn) el.appendChild(drawn);
		else this.term.open(el);
		this.nudge = true;
	}

	/** The ONE writer path: what this view's container measured. */
	propose(cols: number, rows: number): void {
		// Consumed only once it really went out: a CONNECTING socket drops the whole first launch.
		if (this.nudge) {
			if (this.resize(Math.max(1, cols - 1), rows)) this.nudge = false;
		} else if (this.said === `${cols}x${rows}`) {
			return;
		}
		this.resize(cols, rows);
	}

	/** Measure and propose — what a container `ResizeObserver` fires, and the ONLY proposer. */
	refit(): void {
		const d = this.term.proposeDimensions();
		if (d) this.propose(d.cols, d.rows);
	}

	/** Give up this view's say in the size on unmount; the socket stays, so the transcript arrives. */
	retract(): void {
		this.resize(0, 0);
	}

	/** The user's explicit Detach: close the socket, keep the terminal and its scrollback. */
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

/** Drop this instance's VIEW, wherever the Detach was asked; the terminal stays. */
export function detachTermSession(id: string): void {
	live.get(id)?.detach();
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
