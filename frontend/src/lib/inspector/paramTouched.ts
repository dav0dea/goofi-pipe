/** Which params a reader has actually touched, so a plugin's hundreds can collapse to the few in play. */
import type { ParamDescriptor } from '$lib/api/types';

/**
 * Whether a param has been touched: driven by something, or moved off its declared default.
 * DERIVED rather than recorded — a knob turned in a plugin's own window enters the document by the
 * same param door as any other author, so its value alone already says it was touched, and there is
 * no second record of "which knobs were turned" to keep in step.
 */
export function isModified(d: ParamDescriptor): boolean {
	if (d.mode !== 'constant') return true;
	if (d.type === 'pulse') return false;
	return d.value !== d.default;
}

/** How many of a node's params are touched, across every group. */
export function touchedCount(groups: Record<string, Record<string, ParamDescriptor>> | undefined): number {
	return Object.values(groups ?? {}).reduce(
		(n, named) => n + Object.values(named ?? {}).filter(isModified).length,
		0
	);
}
