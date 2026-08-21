/** Copy text to the clipboard. The Clipboard API is absent outside a secure context (plain http on
 * a LAN), so fall back to the hidden-textarea `execCommand` path, which needs a user gesture. */
export async function copyText(text: string): Promise<boolean> {
	try {
		if (navigator.clipboard?.writeText) {
			await navigator.clipboard.writeText(text);
			return true;
		}
	} catch {
	}
	return legacyCopy(text);
}

function legacyCopy(text: string): boolean {
	try {
		const ta = document.createElement('textarea');
		ta.value = text;
		ta.setAttribute('readonly', '');
		ta.style.position = 'fixed';
		ta.style.top = '-9999px';
		ta.style.opacity = '0';
		document.body.appendChild(ta);
		ta.select();
		const ok = document.execCommand('copy');
		document.body.removeChild(ta);
		return ok;
	} catch {
		return false;
	}
}
