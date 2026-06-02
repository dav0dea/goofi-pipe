/**
 * Copy text to the system clipboard.
 *
 * `navigator.clipboard` only exists in a *secure context* — https, or http on
 * `localhost`/`127.0.0.1`. goofi-pipe is usually opened over plain http on a LAN
 * address (`http://<host>:8000`), where the Clipboard API is absent and
 * `navigator.clipboard.writeText` throws. Fall back to the legacy
 * hidden-textarea + `execCommand('copy')` path, which works in any context as
 * long as it runs inside a user gesture (a click handler).
 *
 * Returns whether the copy succeeded.
 */
export async function copyText(text: string): Promise<boolean> {
	try {
		if (navigator.clipboard?.writeText) {
			await navigator.clipboard.writeText(text);
			return true;
		}
	} catch {
		// Permission/context refusal — fall through to the legacy path.
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
