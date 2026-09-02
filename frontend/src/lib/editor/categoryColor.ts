/** Map a goofi DataType (ARRAY/STRING/TABLE/…) to a CSS color variable. */
export function dtypeColor(dtype: string | undefined | null): string {
	switch ((dtype ?? '').toUpperCase()) {
		case 'ARRAY':
			return 'var(--dtype-array)';
		case 'STRING':
			return 'var(--dtype-string)';
		case 'TABLE':
			return 'var(--dtype-table)';
		case 'AUDIO':
			return 'var(--dtype-audio)';
		default:
			return 'var(--text-muted)';
	}
}

export function formatName(s: string): string {
	return s.replace(/_/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase());
}
