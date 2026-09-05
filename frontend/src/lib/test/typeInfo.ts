import type { NodeTypeInfo } from '$lib/api/control';

/** A palette row with every field the client reads, overridden where a test cares. */
export function typeInfo(over: Partial<NodeTypeInfo> = {}): NodeTypeInfo {
	return {
		type: 'Thing',
		tags: [],
		doc: '',
		available: true,
		source: 'builtin',
		missing_deps: [],
		input_slots: {},
		output_slots: {},
		params: {},
		...over
	};
}
