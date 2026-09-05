/** The `engine:Name` type id. A structural type has no engine and stays bare. */
export function bareName(type: string): string {
	const i = type.indexOf(':');
	return i < 0 ? type : type.slice(i + 1);
}

export function engineOf(type: string): string | null {
	const i = type.indexOf(':');
	return i < 0 ? null : type.slice(0, i);
}
