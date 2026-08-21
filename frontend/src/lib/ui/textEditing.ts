/** The one answer both keyboard scopes ask: does this keystroke belong to a text editor or to the
 * app? Duck-typed over the two fields it reads, so it needs no DOM. */
export function isTextEditingTarget(target: EventTarget | null): boolean {
	const el = target as (Partial<HTMLElement> & EventTarget) | null;
	if (!el) return false;
	if (el.isContentEditable === true) return true;
	const tag = el.tagName ?? '';
	return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
}
