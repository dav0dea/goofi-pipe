<!--
  ExprEditor — the param source surface, in one of two configurations: a Python expression with
  goofi's completions, or — given `picker` — a bare field whose only legal contents are the names
  the picker hands it. It reads the graph store, so it lives here and never in `$lib/ui`, which
  must stay a leaf layer.
-->
<script lang="ts">
	import { loadExprEditor } from './load';
	import type { ExprEditorHandle } from './editor';
	import { liveCatalogue } from './catalogue';
	import type { PickerOption } from './refs';

	let {
		value,
		error = null,
		onCommit,
		label,
		placeholder = '',
		testid,
		selfName,
		picker,
		disabled = false
	}: {
		value: string;
		error?: string | null;
		onCommit: (value: string) => void;
		/** The editable element's accessible name. */
		label: string;
		placeholder?: string;
		/** Lands on the editable element. */
		testid: string;
		/** The edited node's display name — what `me` completes against. */
		selfName?: string;
		/** The picker configuration: the names offered, read at the moment the list opens. */
		picker?: () => PickerOption[];
		disabled?: boolean;
	} = $props();

	let host = $state<HTMLDivElement | null>(null);
	let handle = $state<ExprEditorHandle | null>(null);

	/* Everything is read inside the `then`, off the tracking pass, so a keystroke echo does not
	   remount the editor. */
	$effect(() => {
		const el = host;
		if (!el) return;
		let live = true;
		let mounted: ExprEditorHandle | null = null;
		loadExprEditor().then((mod) => {
			if (!live) return;
			const attributes = { 'data-testid': testid, 'aria-label': label };
			mounted = picker
				? mod.createPicker(el, { doc: value, options: picker, onCommit: (v) => onCommit(v), placeholder, attributes })
				: mod.createExprEditor(el, {
						doc: value,
						catalogue: () => ({ ...liveCatalogue(), self: selfName }),
						onCommit: (v) => onCommit(v),
						error,
						placeholder,
						attributes
					});
			handle = mounted;
		});
		return () => {
			live = false;
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

<div class="expr-host" class:disabled bind:this={host} inert={disabled || undefined}>
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
	.expr-host.disabled {
		opacity: 0.5;
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
