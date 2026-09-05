/** Descriptor → the inspector control that renders it. First match wins. */
import type { ParamDescriptor } from '$lib/api/types';

export type ControlKind =
	| 'pulse'
	| 'expression'
	| 'reference'
	| 'numeric'
	| 'toggle'
	| 'select'
	| 'text'
	| 'unknown';

export function controlKind(descriptor: ParamDescriptor): ControlKind {
	if (descriptor.type === 'pulse') return 'pulse';
	if (descriptor.mode === 'expression') return 'expression';
	if (descriptor.mode === 'reference') return 'reference';
	switch (descriptor.type) {
		case 'float':
		case 'int':
			return 'numeric';
		case 'bool':
			return 'toggle';
		case 'string':
			return (descriptor.options?.length ?? 0) > 0 || descriptor.refreshable ? 'select' : 'text';
		default:
			return 'unknown';
	}
}
