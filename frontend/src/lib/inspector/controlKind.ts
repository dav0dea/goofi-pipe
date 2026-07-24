/**
 * Pure descriptor → control discriminant for the inspector (spec §2, D-N2).
 *
 * `ParamField` renders one P `<Field>` whose control region is chosen by this first-match mapping;
 * keeping the decision pure + unit-tested makes the component a thin switch and the mapping one
 * SSOT — never re-derived (and drifting) inside the render path.
 *
 * First match wins (order matters):
 *   1. `expression_enabled`            → 'expression'  (fx editor takes over, whatever the type)
 *   2. `float` | `int`                 → 'numeric'
 *   3. `bool` && `trigger`             → 'trigger'
 *   4. `bool`                          → 'toggle'
 *   5. `string` && (has options || `refreshable`) → 'select'
 *   6. `string`                        → 'text'
 *   7. else                            → 'unknown'
 *
 * The string/select branch keys on `descriptor.refreshable` (NOT the `onRefresh` prop the assembler
 * passes unconditionally): a refreshable string with an EMPTY options list is still a select so its
 * ⟳ re-scan survives.
 */
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
