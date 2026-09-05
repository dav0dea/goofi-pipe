/** The `engine:Name` type id. A structural type has no engine and stays bare. */
export function bareName(type: string): string {
	const i = type.indexOf(':');
	return i < 0 ? type : type.slice(i + 1);
}

export function engineOf(type: string): string | null {
	const i = type.indexOf(':');
	return i < 0 ? null : type.slice(0, i);
}

/** One palette family's ink — an engine, or a plugin format. No family, or no token, reads muted. */
export function familyColor(family: string | null): string {
	return family ? `var(--engine-${family}, var(--text-muted))` : 'var(--text-muted)';
}

/** The engine's ink, for a canvas node, which knows its type and not where it was found. */
export function engineColor(type: string): string {
	return familyColor(engineOf(type));
}
