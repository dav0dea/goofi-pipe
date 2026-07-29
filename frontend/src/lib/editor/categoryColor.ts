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
			return 'var(--text-muted)';
	}
}

export function formatName(s: string): string {
	return s.replace(/_/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase());
}
