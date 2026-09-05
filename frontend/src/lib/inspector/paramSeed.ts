/** A param's current value as the Python literal that seeds its expression. */
import type { ParamDescriptor } from '$lib/api/types';

export function literalFor(d: ParamDescriptor): string {
	if (d.type === 'pulse') return 'False';
	const v = d.value;
	if (typeof v === 'number') return String(v);
	if (typeof v === 'boolean') return v ? 'True' : 'False';
	return JSON.stringify(v);
}
