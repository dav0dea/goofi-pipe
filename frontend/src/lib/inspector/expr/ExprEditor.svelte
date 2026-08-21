<!--
  ExprEditor — the param expression surface, in one of two modes. It reads the graph store, so it lives
  here and never in `$lib/ui`, which must stay a leaf layer.
-->
<script lang="ts">
	import { loadExprEditor } from './load';
	import type { ExprEditorHandle } from './editor';
	import { liveCatalogue } from './catalogue';

	let {
		value,
		multiline = false,
		error = null,
		onCommit,
		onCancel = () => {},
		label,
		placeholder = '',
		testid,
		autofocus = false,
		bindCommit
	}: {
		value: string;
		multiline?: boolean;
		error?: string | null;
		onCommit: (value: string) => void;
		onCancel?: () => void;
		/** The editable element's accessible name. */
		label: string;
		placeholder?: string;
		/** Lands on the editable element. */
		testid: string;
		autofocus?: boolean;
		/** Hands the owner a commit-now function (and `null` on teardown). */
		bindCommit?: (commit: (() => void) | null) => void;
	} = $props();

	let host = $state<HTMLDivElement | null>(null);
	let handle = $state<ExprEditorHandle | null>(null);

	/* `multiline` is read SYNCHRONOUSLY so a mode flip rebuilds the editor; everything else is read
	   inside the `then`, off the tracking pass, so a keystroke echo does not remount it. */
	$effect(() => {
		const el = host;
		const mode = multiline;
		if (!el) return;
		let live = true;
		let mounted: ExprEditorHandle | null = null;
		loadExprEditor().then((mod) => {
			if (!live) return;
			mounted = mod.createExprEditor(el, {
				doc: value,
				multiline: mode,
				catalogue: liveCatalogue,
				onCommit: (v) => onCommit(v),
				onCancel: () => onCancel(),
				error,
				placeholder,
				attributes: { 'data-testid': testid, 'aria-label': label }
			});
			handle = mounted;
			bindCommit?.(() => mounted?.commit());
			if (autofocus) mounted.focus();
		});
		return () => {
			live = false;
			bindCommit?.(null);
			mounted?.destroy();
			handle = null;
		};
	});

	// The handle decides whether adopting the value is safe; it will not interrupt live typing.
	$effect(() => {
		handle?.setValue(value);
	});
	$effect(() => {
		handle?.setError(error);
	});
</script>

<div class="expr-host" class:multi={multiline} bind:this={host}>
	{#if !handle}
		<!-- The stand-in until the lazy chunk lands: out of flow, so the host's height never depends on it. -->
		<pre class="stand-in" aria-hidden="true">{value}</pre>
	{/if}
</div>

<style>
	/* One line's box, before CodeMirror is in it and after: `--lh-text` plus `.cm-content`'s padding
	   (theme.ts) plus the two hairlines. The inline field must never grow. */
	.expr-host {
		position: relative;
		flex: 1;
		min-width: 0;
		min-height: calc(var(--lh-text) * 1em + var(--space-2) * 2 + 2px);
		/* `theme.ts` gives `.cm-scroller` a `fontFamily: 'inherit'`, so this host is where the editor's
		   face is stated — and the stand-in <pre> inherits it too. */
		font-family: var(--font-mono);
		background: var(--surface-1);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		overflow: hidden;
		tab-size: 4;
	}
	.expr-host.multi {
		min-height: 7rem;
		resize: vertical;
		background: var(--surface-2);
	}
	.expr-host.multi > :global(.cm-editor) {
		height: 100%;
	}
	/* Restated because CodeMirror's base theme pins `outline: none` at a specificity app.css cannot reach. */
	.expr-host:has(:focus-visible) {
		outline: var(--focus-width) solid var(--focus-ink);
		outline-offset: 1px;
	}
	.stand-in {
		position: absolute;
		inset: 0;
		margin: 0;
		padding: var(--space-2) var(--space-4);
		overflow: hidden;
		white-space: pre;
		color: var(--text);
		pointer-events: none;
	}
	/* The editable element is a contenteditable div, which app.css's coarse `input` floor cannot reach. */
	@media (hover: none) and (pointer: coarse) {
		.expr-host {
			font-size: 16px;
		}
	}
</style>
