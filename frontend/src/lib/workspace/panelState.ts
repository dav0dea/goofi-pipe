/**
 * Helpers for the opaque per-panel `state` blob.
 *
 * Panel `state` is an untyped bag persisted in the layout tree. Two conventions
 * ride on top of it that several panels — and the framework's node-link cleanup
 * — share, so they live here in one place instead of being re-derived per panel:
 *
 *  - it coerces to a plain object (`asStateObject`)
 *  - node-linked panels (Parameters / Viewer / Metadata) store the bound node's
 *    name under `node` (`linkedNodeName` / `withLinkedNode`)
 *
 * Keeping the `node` key in one module means the model, the workspace store, and
 * every linkable panel agree on where the bound node lives.
 */

/** Coerce opaque panel state to a plain object bag (empty when unset/non-object). */
export function asStateObject(state: unknown): Record<string, unknown> {
	return typeof state === 'object' && state !== null ? (state as Record<string, unknown>) : {};
}

/** The node name a linkable panel is bound to, or null. */
export function linkedNodeName(state: unknown): string | null {
	const v = asStateObject(state).node;
	return typeof v === 'string' ? v : null;
}

/** Panel state with the linked node set (pass null to unlink), preserving the
 * rest of the bag (slot / kind / group). */
export function withLinkedNode(state: unknown, node: string | null): Record<string, unknown> {
	return { ...asStateObject(state), node };
}
