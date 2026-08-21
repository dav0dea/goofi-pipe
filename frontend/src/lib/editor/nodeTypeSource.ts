/** The one word a palette row carries beside a node's name: unavailability, else provenance. */
import type { NodeTypeInfo } from '$lib/api/control';

export function nodeTypeSource(t: NodeTypeInfo): string {
	if (!t.available) return 'unavailable';
	return t.source === 'patch' ? 'this patch' : 'builtin';
}
