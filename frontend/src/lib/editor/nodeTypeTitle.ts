/** The add-menu's tooltip for one palette entry.
 *
 * Always the backend's `doc` — including for an UNAVAILABLE node, where the bridge
 * already phrases the failure ("This node could not be loaded: {reason}"). The client
 * used to re-phrase that as "missing dependency: {reason}", which is only true for a
 * ModuleNotFoundError: for anything else `reason` is the exception line, so a node with
 * a syntax error read "missing dependency: SyntaxError: expected ':'".
 *
 * Lives here rather than inside `AddNodeMenu.svelte` so that decision is unit-testable.
 */
import type { NodeTypeInfo } from '$lib/api/control';

export function nodeTypeTitle(t: NodeTypeInfo): string {
	return t.doc;
}
