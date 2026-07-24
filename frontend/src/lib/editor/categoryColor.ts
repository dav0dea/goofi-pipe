/** Node categories no longer drive colour (design overhaul D4). Every node/dot renders
 * neutral; category still groups the palette, just without a hue. Kept as the single
 * call site so ParamPanel/AddNodeMenu dots neutralise with no markup change. */
export function categoryColor(_category: string | undefined | null): string {
	return 'var(--text-muted)';
}

/** Map a goofi DataType (ARRAY/STRING/TABLE/…) to a CSS color variable.
 *
 * Communicates dtype at a glance on every slot pin/label. Palette borrowed
 * from goofi3 (array=green, string=yellow, table=orange) so the visual
 * grammar is consistent across the two projects.
 */
export function dtypeColor(dtype: string | undefined | null): string {
	switch ((dtype ?? '').toUpperCase()) {
		case 'ARRAY':
			return 'var(--dtype-array)';
		case 'STRING':
			return 'var(--dtype-string)';
		case 'TABLE':
			return 'var(--dtype-table)';
		default:
			return 'var(--text-faint)';
	}
}

export function formatName(s: string): string {
	return s.replace(/_/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase());
}
