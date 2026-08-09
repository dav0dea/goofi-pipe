<script lang="ts">
	import type { DataFrame } from '$lib/codec/decode';
	import type { SettingsMap } from './settingsSchema';
	import { renderMarkdown } from './renderMarkdown';

	type Props = { frame: DataFrame; settings?: SettingsMap };
	const { frame, settings = {} }: Props = $props();

	const useMarkdown = $derived(Boolean(settings.markdown));
	const wrap = $derived(settings.wrap !== false);
	const text = $derived(typeof frame.data === 'string' ? frame.data : String(frame.data));
	// Node-sourced strings are untrusted — renderMarkdown escapes raw HTML and
	// blocks dangerous URL schemes before {@html} (report B14).
	const html = $derived(useMarkdown ? renderMarkdown(text) : '');
</script>

{#if useMarkdown}
	<div class="md" data-testid="string-viewer">{@html html}</div>
{:else}
	<pre class="text" class:nowrap={!wrap} data-testid="string-viewer">{text}</pre>
{/if}

<style>
	/* String Data is a payload, so mono (D-T3) — the same face the node canvas paints it in, and
	   stated here rather than inherited: `.text` is a bare <pre>, which app.css resets to `font:
	   inherit`, and the panel around it is chrome. The markdown body inherits this too, including
	   its `code`/`pre` runs (same reset), so the document reads in one face. */
	.text,
	.md {
		width: 100%;
		height: 100%;
		min-height: 80px;
		font-family: var(--font-mono);
		font-size: var(--fs-small);
		color: var(--text);
		background: var(--surface-1);
		margin: 0;
		padding: var(--space-3) var(--space-5);
		border-radius: var(--radius-sm);
		overflow: auto;
		box-sizing: border-box;
	}
	.text {
		white-space: pre-wrap;
		word-break: break-word;
	}
	.text.nowrap {
		white-space: pre;
		word-break: normal;
	}
	.md {
		line-height: 1.5;
	}
	/* The rendered document's own type scale is deliberately `em`-relative: headings, lists and
	   quotes size against the viewer root (which IS on --fs-*), so the prose keeps its internal
	   hierarchy at every root size. Absolute rungs here would flatten it. */
	.md :global(h1),
	.md :global(h2),
	.md :global(h3) {
		margin: 0.4em 0 0.3em;
		font-size: 1.15em;
		color: var(--text);
	}
	.md :global(h2) {
		font-size: 1.08em;
	}
	.md :global(h3) {
		font-size: 1em;
	}
	.md :global(p) {
		margin: 0.4em 0;
	}
	.md :global(ul),
	.md :global(ol) {
		margin: 0.3em 0;
		padding-left: 1.3em;
	}
	.md :global(code) {
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		padding: 0 var(--space-2);
	}
	.md :global(pre) {
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		padding: var(--space-3) var(--space-5);
		overflow: auto;
	}
	.md :global(pre code) {
		border: 0;
		padding: 0;
	}
	.md :global(a) {
		color: var(--accent);
	}
	.md :global(blockquote) {
		margin: 0.4em 0;
		padding-left: 0.8em;
		border-left: 2px solid var(--border-strong);
		color: var(--text-dim);
	}
</style>
