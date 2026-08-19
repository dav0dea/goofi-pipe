/**
 * Data-plane Web Worker (reports A11/A12/A13).
 *
 * Owns one WebSocket per (node, slot), keeps only the LATEST raw frame, and on
 * a ~display-rate tick decodes just that survivor and posts the decoded frame to
 * the main thread — transferring the array buffer (zero-copy). This moves the WS
 * receive + decode (a per-frame ArrayBuffer copy, ~180 MB/s for HD video) off
 * the main thread, and decodes ~60 frames/s/slot instead of every kHz frame.
 */
import { decodeData, type DataFrame } from '$lib/codec/decode';
import { dataUrl } from './dataUrl';
import { streamKey } from './streamKey';
import { DemandTicker } from './demandTicker';

interface SlotState {
	node: string;
	slot: string;
	ws: WebSocket | null;
	url: string;
	closed: boolean;
	reconnectMs: number;
	latestRaw: ArrayBuffer | null;
	/** The ViewSpecs every viewer of this slot has contributed. Sent inband on
	 * (re)connect and whenever they change; the bridge merges them against the real
	 * frame and reduces the slot to their union. Empty until a viewer reports its
	 * capacity (the bridge sends full-resolution frames until then). */
	specs: unknown[];
}

const slots = new Map<string, SlotState>();
const TICK_MS = 16; // ~60 Hz; workers have no requestAnimationFrame


function sendSpecs(st: SlotState): void {
	if (!st.ws || st.ws.readyState !== WebSocket.OPEN) return;
	try {
		st.ws.send(JSON.stringify({ op: 'view', specs: st.specs }));
	} catch {
		// Closed mid-send; the next (re)connect re-sends from st.specs.
	}
}

function openWs(st: SlotState): void {
	if (st.closed) return;
	const ws = new WebSocket(st.url);
	ws.binaryType = 'arraybuffer';
	st.ws = ws;
	ws.addEventListener('open', () => {
		st.reconnectMs = 250;
		sendSpecs(st); // re-send the ViewSpecs on every (re)connect (no server resume)
	});
	ws.addEventListener('message', (e) => {
		if (e.data instanceof ArrayBuffer) st.latestRaw = e.data; // overwrite — latest wins
	});
	ws.addEventListener('close', (e) => {
		st.ws = null;
		if (st.closed) return;
		// Terminal app close codes (4000+) can never resolve — don't reconnect.
		if (e.code >= 4000) {
			st.closed = true;
			return;
		}
		const delay = st.reconnectMs;
		st.reconnectMs = Math.min(st.reconnectMs * 2, 5000);
		setTimeout(() => openWs(st), delay);
	});
}

function collectBuffers(frame: DataFrame, out: Set<ArrayBufferLike>): void {
	const d = frame.data as unknown;
	if (frame.dtype === 'ARRAY') {
		const values = (d as { values?: ArrayLike<number> & { buffer?: ArrayBufferLike } }).values;
		if (values?.buffer) out.add(values.buffer);
	} else if (frame.dtype === 'TABLE' && d && typeof d === 'object') {
		for (const v of Object.values(d as Record<string, DataFrame>)) collectBuffers(v, out);
	}
}

self.addEventListener('message', (e: MessageEvent) => {
	const m = e.data as { op: string; node: string; slot: string; specs?: unknown[] };
	const k = streamKey(m.node, m.slot);
	// One owner decides: the viewer registry in `frames.ts` sends exactly one 'sub' per stream and
	// one 'unsub' to end it, so there is no count to keep here. A second opinion on how many
	// viewers a slot has is what this file used to hold, and what it must not hold again.
	if (m.op === 'sub') {
		let st = slots.get(k);
		if (!st) {
			const proto = self.location.protocol === 'https:' ? 'wss:' : 'ws:';
			const url = dataUrl(proto, self.location.host, m.node, m.slot);
			st = { node: m.node, slot: m.slot, ws: null, url, closed: false, reconnectMs: 250, latestRaw: null, specs: [] };
			slots.set(k, st);
			syncTicker();
			openWs(st);
		}
	} else if (m.op === 'spec') {
		// Viewers reported (or updated) the ViewSpecs for this slot. The 'sub' for a
		// slot is always posted before any 'spec' (ViewerFeed subscribes in a
		// source-earlier effect, and every spec post is microtask-deferred), so a spec
		// for an absent slot is only a post-unsub straggler — drop it.
		const st = slots.get(k);
		if (st) {
			st.specs = m.specs ?? [];
			sendSpecs(st);
		}
	} else if (m.op === 'unsub') {
		const st = slots.get(k);
		if (!st) return;
		st.closed = true;
		st.ws?.close();
		slots.delete(k);
		syncTicker();
	}
});

/** Drain every slot's latest-wins frame to the main thread. Armed only while slots exist. */
function drain(): void {
	for (const st of slots.values()) {
		const raw = st.latestRaw;
		if (!raw) continue;
		st.latestRaw = null;
		let frame: DataFrame;
		try {
			frame = decodeData(raw);
		} catch {
			continue; // a corrupt frame shouldn't kill the slot
		}
		const transfer = new Set<ArrayBufferLike>();
		collectBuffers(frame, transfer);
		(self as unknown as Worker).postMessage(
			{ node: st.node, slot: st.slot, frame },
			Array.from(transfer) as Transferable[]
		);
	}
}

/** The decode ticker services SUBSCRIBED slots, so it runs only while there are some. Call after
 * every `slots` insert/delete; `DemandTicker.sync` is idempotent. */
const ticker = new DemandTicker(drain, TICK_MS);
function syncTicker(): void {
	ticker.sync(slots.size);
}
