/** Data-plane Web Worker: one WebSocket per (node, slot), latest-wins, decoded on a tick and
 * posted to the main thread with its array buffer transferred. */
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
	/** The ViewSpecs every viewer of this slot has contributed; empty means full resolution. */
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
		sendSpecs(st); // no server resume
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
		// A 'sub' always precedes a 'spec', so a spec for an absent slot is a post-unsub straggler.
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

/** Call after every `slots` insert/delete; `DemandTicker.sync` is idempotent. */
const ticker = new DemandTicker(drain, TICK_MS);
function syncTicker(): void {
	ticker.sync(slots.size);
}
