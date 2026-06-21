import { describe, it, expect } from 'vitest';
import { renderMarkdown } from './renderMarkdown';

// Report B14: StringViewer renders node-sourced text as markdown. A node can
// emit arbitrary strings, so the markdown→HTML path must neutralize injected
// markup and dangerous URL schemes while preserving normal formatting.
describe('renderMarkdown', () => {
	it('escapes raw HTML so injected event handlers cannot fire', () => {
		const out = renderMarkdown('<img src=x onerror=alert(1)>');
		// The tag is escaped to inert text — no live <img> element reaches the DOM.
		expect(out).not.toContain('<img');
		expect(out).toContain('&lt;img');
	});

	it('escapes <script> tags', () => {
		const out = renderMarkdown('hi <script>alert(1)</script>');
		expect(out).not.toContain('<script>');
		expect(out).not.toContain('</script>');
	});

	it('neutralizes javascript: URLs in links', () => {
		const out = renderMarkdown('[click](javascript:alert(1))');
		expect(out.toLowerCase()).not.toContain('javascript:');
		// The visible link text is preserved.
		expect(out).toContain('click');
	});

	it('neutralizes javascript: URLs in images', () => {
		const out = renderMarkdown('![x](javascript:alert(1))');
		expect(out.toLowerCase()).not.toContain('javascript:');
	});

	it('preserves normal markdown formatting', () => {
		const out = renderMarkdown('# Title\n\n**bold** and `code`\n\n- item');
		expect(out).toContain('<h1');
		expect(out).toContain('<strong>bold</strong>');
		expect(out).toContain('<code>code</code>');
		expect(out).toContain('<li>item</li>');
	});

	it('keeps safe http(s) links', () => {
		const out = renderMarkdown('[goofi](https://example.com)');
		expect(out).toContain('href="https://example.com"');
	});
});
