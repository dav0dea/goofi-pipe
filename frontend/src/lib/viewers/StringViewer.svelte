<script lang="ts">
	import type { DataFrame } from '$lib/codec/decode';
	import type { SettingsMap } from './viewerSettings.svelte';
	import { marked } from 'marked';

	type Props = { frame: DataFrame; settings?: SettingsMap };
	const { frame, settings = {} }: Props = $props();

	const useMarkdown = $derived(Boolean(settings.markdown));
	const wrap = $derived(settings.wrap !== false);
	const text = $derived(typeof frame.data === 'string' ? frame.data : String(frame.data));
	// Local single-user tool rendering a node's own string output — default
	// marked (inline HTML allowed) is acceptable here.
	const html = $derived(useMarkdown ? (marked.parse(text) as string) : '');
</script>

{#if useMarkdown}
	<div class="md" data-testid="string-viewer">{@html html}</div>
{:else}
	<pre class="text" class:nowrap={!wrap} data-testid="string-viewer">{text}</pre>
{/if}

<style>
	.text,
	.md {
		width: 100%;
		height: 100%;
		min-height: 80px;
		font-family: var(--font-mono);
		font-size: 11px;
		color: var(--text);
		background: var(--bg-elev-1);
		margin: 0;
		padding: 6px 8px;
		border-radius: 4px;
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
		border-radius: 3px;
		padding: 0 3px;
	}
	.md :global(pre) {
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: 4px;
		padding: 6px 8px;
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
