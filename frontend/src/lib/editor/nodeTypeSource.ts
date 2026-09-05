/** The one word a palette row carries beside a node's name: unavailability, else provenance. */
import type { NodeTypeInfo } from '$lib/api/control';

const WORD = { patch: 'this patch', plugin: 'plugin', builtin: 'builtin' } as const;

export function nodeTypeSource(t: NodeTypeInfo): string {
	return t.available ? WORD[t.source] : 'unavailable';
}
