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

/** One row of the touched list: the param, and the group it had to be fetched out of. */
export interface TouchedRow {
	group: string;
	name: string;
	descriptor: ParamDescriptor;
}

/**
 * Every touched param, ACROSS the groups. Spanning them is the whole point: a knob was turned in
 * the plugin's own window, and which tab goofi filed it under is the one thing the reader does not
 * know — a per-tab filter answers "nothing here" while the count says otherwise.
 */
export function touchedRows(
	groups: Record<string, Record<string, ParamDescriptor>> | undefined,
	order?: string[]
): TouchedRow[] {
	const names = order ?? Object.keys(groups ?? {});
	return names.flatMap((group) =>
		Object.entries(groups?.[group] ?? {})
			.filter(([, descriptor]) => isModified(descriptor))
			.map(([name, descriptor]) => ({ group, name, descriptor }))
	);
}

/** How many of a node's params are touched, across every group. */
export function touchedCount(groups: Record<string, Record<string, ParamDescriptor>> | undefined): number {
	return Object.values(groups ?? {}).reduce(
		(n, named) => n + Object.values(named ?? {}).filter(isModified).length,
		0
	);
}
