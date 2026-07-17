/**
 * Browser half of the CRDT sync protocol — the exact mirror of the Rust
 * `goofi_crdt::SyncMsg` + `GraphDoc::on_sync` (see the crate). Two framed message kinds,
 * one leading tag byte: a peer advertises what it has (`StateVector`) and ships what the
 * other lacks (`Update`). Both ends drive their doc by hand via the Yjs primitives
 * (`encodeStateVector` / `encodeStateAsUpdate` / `applyUpdate`), so no `y-protocols`
 * dependency is needed — this pure module IS the protocol.
 *
 * Kept free of any WebSocket/Svelte glue so it is unit-testable against a real `Y.Doc`.
 */
import * as Y from 'yjs';

/** Tag bytes — must match `SYNC_TAG_*` in the Rust crate. */
export const SYNC_TAG_SV = 0;
export const SYNC_TAG_UPDATE = 1;
export const SYNC_TAG_EPHEMERAL = 2;

export type SyncMsg =
	| { kind: 'sv'; payload: Uint8Array }
	| { kind: 'update'; payload: Uint8Array }
	| { kind: 'ephemeral'; payload: Uint8Array };

const TAG_BY_KIND = { sv: SYNC_TAG_SV, update: SYNC_TAG_UPDATE, ephemeral: SYNC_TAG_EPHEMERAL };

/** Frame a message as `[tag, ...payload]`. */
export function encodeSyncMsg(msg: SyncMsg): Uint8Array {
	const out = new Uint8Array(msg.payload.length + 1);
	out[0] = TAG_BY_KIND[msg.kind];
	out.set(msg.payload, 1);
	return out;
}

/** Parse a framed message; `null` on empty input or an unknown tag. */
export function decodeSyncMsg(bytes: Uint8Array): SyncMsg | null {
	if (bytes.length === 0) return null;
	const payload = bytes.subarray(1);
	if (bytes[0] === SYNC_TAG_SV) return { kind: 'sv', payload };
	if (bytes[0] === SYNC_TAG_UPDATE) return { kind: 'update', payload };
	if (bytes[0] === SYNC_TAG_EPHEMERAL) return { kind: 'ephemeral', payload };
	return null;
}

/**
 * The message to send a peer on connect: this doc's state vector, framed. The peer
 * answers with the diff it owes (via {@link onSync}).
 */
export function syncHello(doc: Y.Doc): Uint8Array {
	return encodeSyncMsg({ kind: 'sv', payload: Y.encodeStateVector(doc) });
}

/**
 * Drive the pairwise sync handshake for one inbound message, returning the frames to send
 * back. Receiving a peer's `StateVector` yields the `Update` it lacks; receiving an
 * `Update` applies it and replies with nothing. Symmetric with the Rust `on_sync` — both
 * ends run this. `origin` tags the transaction so the app can distinguish remote (synced)
 * changes from its own local edits (used by the reactive bridge + undo scoping later).
 */
export function onSync(doc: Y.Doc, msg: SyncMsg, origin: unknown = 'remote'): Uint8Array[] {
	if (msg.kind === 'sv') {
		return [encodeSyncMsg({ kind: 'update', payload: Y.encodeStateAsUpdate(doc, msg.payload) })];
	}
	if (msg.kind === 'update') {
		// Applying with a distinct origin lets observers ignore remote echoes of their own writes.
		Y.applyUpdate(doc, msg.payload, origin);
	}
	// An ephemeral frame is never a doc update — it is routed to the EphemeralStore, not here.
	return [];
}
