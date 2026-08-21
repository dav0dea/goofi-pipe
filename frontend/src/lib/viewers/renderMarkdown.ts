import { Marked, type Tokens } from 'marked';

// Node strings are untrusted: `marked` itself is hardened, since DOMPurify needs a DOM.

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
		html({ text }: Tokens.HTML | Tokens.Tag): string {
			return escapeHtml(text);
		},
		link(token: Tokens.Link): string {
			const inner = this.parser.parseInline(token.tokens);
			if (!isSafeHref(token.href)) {
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
