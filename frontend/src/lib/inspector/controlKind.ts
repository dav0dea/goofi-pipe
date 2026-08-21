/** Descriptor → the inspector control that renders it. First match wins. */
import type { ParamDescriptor } from '$lib/api/types';

export type ControlKind =
	| 'expression'
	| 'numeric'
	| 'trigger'
	| 'toggle'
	| 'select'
	| 'text'
	| 'unknown';

export function controlKind(descriptor: ParamDescriptor): ControlKind {
	if (descriptor.expression_enabled) return 'expression';
	switch (descriptor.type) {
		case 'float':
		case 'int':
			return 'numeric';
		case 'bool':
			return descriptor.trigger ? 'trigger' : 'toggle';
		case 'string':
			return (descriptor.options?.length ?? 0) > 0 || descriptor.refreshable ? 'select' : 'text';
		default:
			return 'unknown';
	}
}
