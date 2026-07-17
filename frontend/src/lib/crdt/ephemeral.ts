/**
 * The ephemeral / awareness channel — presence-style state (cursors, selection, live-drag
 * values, expression previews, active viewer specs) that rides the `/control` binary channel
 * but is NEVER written to the doc and NEVER recovered on lag. The manager relays each frame
 * verbatim to all clients; peers self-filter their own `client` id.
 *
 * Frame layout inside `SyncMsg::Ephemeral`'s payload (browser-owned; the manager is opaque):
 *   [ client: u64 LE (8 bytes) | state: UTF-8 JSON ]
 * An empty state (0 remaining bytes) is a REMOVAL — the client left / cleared its presence.
 *
 * Pure of Svelte/WebSocket, so it unit-tests directly. The reactive layer wraps a store.
 */

export interface EphemeralUpdate {
	client: number;
	/** The client's ephemeral state, or `null` to remove its entry (departure/clear). */
	state: unknown | null;
}

/** Encode an ephemeral update into the payload carried by `SyncMsg::Ephemeral`. */
export function encodeEphemeral(update: EphemeralUpdate): Uint8Array {
	const body = update.state === null ? new Uint8Array(0) : utf8(JSON.stringify(update.state));
	const out = new Uint8Array(8 + body.length);
	const view = new DataView(out.buffer);
	view.setBigUint64(0, BigInt(update.client >>> 0), true); // LE, matches Rust u64 LE
	out.set(body, 8);
	return out;
}

/** Decode an ephemeral payload; `null` if it is too short or the state JSON is malformed. */
export function decodeEphemeral(bytes: Uint8Array): EphemeralUpdate | null {
	if (bytes.length < 8) return null;
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	const client = Number(view.getBigUint64(0, true));
	if (bytes.length === 8) return { client, state: null }; // removal
	try {
		return { client, state: JSON.parse(utf8Decode(bytes.subarray(8))) };
	} catch {
		return null;
	}
}

/**
 * Tracks every OTHER client's latest ephemeral state, keyed by client id. Applying an update
 * for `localClient` is a no-op (self-filter), so a peer never sees its own presence echoed.
 * `entries()` is the current remote presence set (for cursors, preview overlays, etc.).
 */
export class EphemeralStore {
	private states = new Map<number, unknown>();
	private localClient: number;

	constructor(localClient: number) {
		this.localClient = localClient;
	}

	/** Apply a relayed ephemeral update. Own-id updates are ignored; a null state removes. */
	apply(update: EphemeralUpdate): void {
		if (update.client === this.localClient) return;
		if (update.state === null) this.states.delete(update.client);
		else this.states.set(update.client, update.state);
	}

	/** Remote clients' current states, as `[client, state]` pairs. */
	entries(): Array<[number, unknown]> {
		return [...this.states.entries()];
	}

	/** The state of one remote client, or `undefined`. */
	get(client: number): unknown | undefined {
		return this.states.get(client);
	}

	get size(): number {
		return this.states.size;
	}
}

function utf8(s: string): Uint8Array {
	return new TextEncoder().encode(s);
}
function utf8Decode(b: Uint8Array): string {
	return new TextDecoder().decode(b);
}
