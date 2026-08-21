/** The add-menu's tooltip: always the backend's `doc`, which phrases an unavailable node too. */
import type { NodeTypeInfo } from '$lib/api/control';

export function nodeTypeTitle(t: NodeTypeInfo): string {
	return t.doc;
}
