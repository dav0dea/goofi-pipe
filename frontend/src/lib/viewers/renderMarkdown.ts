import { Marked, type Tokens } from 'marked';

// Report B14: a node can emit arbitrary strings, so the markdown→HTML path
// must never let node-sourced text inject live markup. We harden `marked`
// itself rather than post-sanitizing with a DOM library (DOMPurify needs a
// DOM, which the unit env lacks). For markdown specifically this is the right
// trade: disallow raw HTML passthrough and block dangerous URL schemes — a
// node's string output loses nothing meaningful and cannot script the page.

const DANGEROUS_SCHEME = /^\s*(?:javascript|data|vbscript):/i;

function escapeHtml(s: string): string {
	return s
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;')
		.replace(/'/g, '&#39;');
}

function isSafeHref(href: string): boolean {
	return !DANGEROUS_SCHEME.test(href);
}

const md = new Marked();
md.use({
	renderer: {
		// Raw HTML tokens (both block `html` and inline `Tag`) are escaped to
		// inert text instead of being passed through verbatim.
		html({ text }: Tokens.HTML | Tokens.Tag): string {
			return escapeHtml(text);
		},
		link(token: Tokens.Link): string {
			const inner = this.parser.parseInline(token.tokens);
			if (!isSafeHref(token.href)) {
				// Drop the unsafe URL but keep the visible link text.
				return inner;
			}
			const title = token.title ? ` title="${escapeHtml(token.title)}"` : '';
			return `<a href="${escapeHtml(token.href)}"${title}>${inner}</a>`;
		},
		image(token: Tokens.Image): string {
			if (!isSafeHref(token.href)) {
				return escapeHtml(token.text);
			}
			const title = token.title ? ` title="${escapeHtml(token.title)}"` : '';
			return `<img src="${escapeHtml(token.href)}" alt="${escapeHtml(token.text)}"${title}>`;
		}
	}
});

/** Render node-sourced markdown to sanitized HTML (pure; no DOM required). */
export function renderMarkdown(text: string): string {
	return md.parse(text, { async: false }) as string;
}
