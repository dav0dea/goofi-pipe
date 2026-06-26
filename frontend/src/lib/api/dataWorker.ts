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

interface SlotState {
	node: string;
	slot: string;
	kind: string;
	ws: WebSocket | null;
	url: string;
	closed: boolean;
	reconnectMs: number;
	latestRaw: ArrayBuffer | null;
	refs: number;
	/** Latest per-axis ViewSpec the viewer wants this slot reduced to (Option C).
	 * Sent inband on (re)connect and whenever it changes; null until the viewer
	 * reports its capacity (the node uses its kind-default until then). */
	spec: unknown | null;
}

const slots = new Map<string, SlotState>();
const TICK_MS = 16; // ~60 Hz; workers have no requestAnimationFrame

function key(node: string, slot: string, kind: string): string {
	return `${node} ${slot} ${kind}`;
}

function sendSpec(st: SlotState): void {
	if (st.spec == null || !st.ws || st.ws.readyState !== WebSocket.OPEN) return;
	try {
		st.ws.send(JSON.stringify({ op: 'view', spec: st.spec }));
	} catch {
		// Closed mid-send; the next (re)connect re-sends from st.spec.
	}
}

function openWs(st: SlotState): void {
	if (st.closed) return;
	const ws = new WebSocket(st.url);
	ws.binaryType = 'arraybuffer';
	st.ws = ws;
	ws.addEventListener('open', () => {
		st.reconnectMs = 250;
		sendSpec(st); // re-send the ViewSpec on every (re)connect (no server resume)
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
	const m = e.data as { op: string; node: string; slot: string; kind: string; spec?: unknown };
	const k = key(m.node, m.slot, m.kind);
	if (m.op === 'sub') {
		let st = slots.get(k);
		if (!st) {
			const proto = self.location.protocol === 'https:' ? 'wss:' : 'ws:';
			const url = dataUrl(proto, self.location.host, m.node, m.slot, m.kind);
			st = { node: m.node, slot: m.slot, kind: m.kind, ws: null, url, closed: false, reconnectMs: 250, latestRaw: null, refs: 0, spec: null };
			slots.set(k, st);
			openWs(st);
		}
		st.refs++;
	} else if (m.op === 'spec') {
		// Viewer reported (or updated) the capacity ViewSpec for this slot.
		const st = slots.get(k);
		if (st) {
			st.spec = m.spec ?? null;
			sendSpec(st);
		}
	} else if (m.op === 'unsub') {
		const st = slots.get(k);
		if (!st) return;
		st.refs--;
		if (st.refs <= 0) {
			st.closed = true;
			st.ws?.close();
			slots.delete(k);
		}
	}
});

setInterval(() => {
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
			{ node: st.node, slot: st.slot, kind: st.kind, frame },
			Array.from(transfer) as Transferable[]
		);
	}
}, TICK_MS);
