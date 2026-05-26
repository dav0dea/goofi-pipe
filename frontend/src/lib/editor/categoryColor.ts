/** Map a node category to a CSS color variable. */
export function categoryColor(category: string | undefined | null): string {
	switch ((category ?? '').toLowerCase()) {
		case 'analysis':
			return 'var(--cat-analysis)';
		case 'array':
			return 'var(--cat-array)';
		case 'inputs':
			return 'var(--cat-inputs)';
		case 'misc':
			return 'var(--cat-misc)';
		case 'outputs':
			return 'var(--cat-outputs)';
		case 'signal':
			return 'var(--cat-signal)';
		default:
			return 'var(--cat-viewer)';
	}
}

export function formatName(s: string): string {
	return s.replace(/_/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase());
}
