/**
 * Helpers for the opaque per-panel `state` blob.
 *
 * Panel `state` is an untyped bag persisted in the layout tree. Two conventions
 * ride on top of it that several panels — and the framework's node-link cleanup
 * — share, so they live here in one place instead of being re-derived per panel:
 *
 *  - it coerces to a plain object (`asStateObject`)
 *  - node-linked panels (Parameters / Viewer / Metadata) store the bound node's
 *    uid (its stable identity, not the display name) under `node`
 *    (`linkedNodeName` / `withLinkedNode` — names kept for compatibility)
 *  - a node editor stores its sub-patch depth under `subpatchPath`, encoded by
 *    `pathToArray` / `arrayToPath`
 *
 * Keeping these keys in one module means the model, the workspace store, and
 * every linkable panel agree on where each one lives.
 */

/** Coerce opaque panel state to a plain object bag (empty when unset/non-object). */
export function asStateObject(state: unknown): Record<string, unknown> {
	return typeof state === 'object' && state !== null ? (state as Record<string, unknown>) : {};
}

/** The node uid (stable identity) a linkable panel is bound to, or null. */
export function linkedNodeName(state: unknown): string | null {
	const v = asStateObject(state).node;
	return typeof v === 'string' ? v : null;
}

/** Panel state with the linked node set (pass null to unlink), preserving the
 * rest of the bag (slot / kind / group). */
export function withLinkedNode(state: unknown, node: string | null): Record<string, unknown> {
	return { ...asStateObject(state), node };
}

// The sub-patch an editor is currently inside, as a path persisted in the
// panel's layout state so save/reload (and same-session reconnect) recover the
// exact view: '/' is the root patch, '/subpatch0' / '/subpatch0/subpatch1'
// descend into nested sub-patches. Written by the panel, read back by the
// nav-context restore — one encoding so the round trip cannot drift.
export function pathToArray(p: unknown): string[] {
	return typeof p === 'string' ? p.split('/').filter(Boolean) : [];
}
export function arrayToPath(a: string[]): string {
	return '/' + a.join('/');
}
