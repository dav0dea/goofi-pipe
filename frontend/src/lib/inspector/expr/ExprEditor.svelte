<!--
  ExprEditor — the param expression surface: Python syntax highlighting, one merged completion popup
  (stock Python + goofi's `nd()` / `globals.` / `np.`), and the backend's compile/eval error as an
  inline diagnostic. ONE component in two modes (D-X1) — the inline param field and the expanded
  in-panel editor are the same editor, differing only in whether the document may hold more than one
  line and what Enter/Escape do.

  It lives in `lib/inspector/expr/`, never in `lib/ui/` (D-X4): the completion source reads the graph
  store and the globals doc root, and `$lib/ui` must not import `$lib/stores` — the primitives are a
  leaf layer, and breaking that reshuffles Vite's CSS chunk graph (it cost the app a flash of unstyled
  chrome on first paint once already). So this is inspector code that USES primitives.

  The editor itself is LAZY (D-X5). Nothing here imports a `@codemirror/*` module; `load.ts` does, once,
  behind an `await import()`, and until it resolves the host renders the current expression as plain
  text at the same geometry — absolutely positioned, so the host's height is its own from the first
  frame and the swap moves nothing.
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
		autofocus = false
	}: {
		value: string;
		multiline?: boolean;
		error?: string | null;
		onCommit: (value: string) => void;
		onCancel?: () => void;
		/** The editable element's accessible name. */
		label: string;
		placeholder?: string;
		/** Lands on the editable element, so a spec drives the editor the way it drove the textarea. */
		testid: string;
		autofocus?: boolean;
	} = $props();

	let host = $state<HTMLDivElement | null>(null);
	let handle = $state<ExprEditorHandle | null>(null);

	/* Mount. `multiline` is read SYNCHRONOUSLY so a mode flip rebuilds the editor by construction;
	   everything else is read inside the `then`, which is off the tracking pass — a keystroke echo must
	   not tear the editor down and build a new one. */
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
			if (autofocus) mounted.focus();
		});
		return () => {
			live = false;
			mounted?.destroy();
			handle = null;
		};
	});

	// An externally changed source / error, adopted once the editor is up. The handle itself decides
	// whether adopting the value is safe — it will not yank the document out from under live typing.
	$effect(() => {
		handle?.setValue(value);
	});
	$effect(() => {
		handle?.setError(error);
	});
</script>

<div class="expr-host" class:multi={multiline} bind:this={host}>
	{#if !handle}
		<!-- The same-geometry stand-in (D-X5): the expression as plain text, out of flow so the host's
		     height never depends on it, and gone the moment the real editor is up. -->
		<pre class="stand-in" aria-hidden="true">{value}</pre>
	{/if}
</div>

<style>
	/* One line's worth of box before CodeMirror is in it, and after: `--lh-text` leading plus
	   `.cm-content`'s vertical padding (theme.ts) plus the two hairlines the border-box includes. The
	   inline field must never GROW (success criterion 1), so long source scrolls sideways inside the
	   scroller rather than wrapping — CodeMirror's default with no `lineWrapping`. */
	.expr-host {
		position: relative;
		flex: 1;
		min-width: 0;
		min-height: calc(var(--lh-text) * 1em + var(--space-2) * 2 + 2px);
		background: var(--surface-1);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		overflow: hidden;
		tab-size: 4;
	}
	/* The expanded editor keeps the multi-line box it had as a textarea, resizable in place (D-N4 —
	   it grows in the panel so the graph stays visible; no modal). */
	.expr-host.multi {
		min-height: 7rem;
		resize: vertical;
		background: var(--surface-2);
	}
	.expr-host.multi > :global(.cm-editor) {
		height: 100%;
	}
	/* The app's own keyboard ring, restated because CodeMirror's base theme pins `outline: none` on the
	   editable element at a specificity app.css's global rule cannot reach. `:has()` keeps the trigger
	   exactly where app.css puts it — the focused control — while the ring draws on the box around it. */
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
	/* Touch: the editable element is a contenteditable div, so app.css's `input, select, textarea`
	   coarse floor cannot reach it — iOS force-zooms on focus below 16px either way. Inherited by
	   `.cm-content`, which is what `touch-inspector.spec.ts` measures. */
	@media (hover: none) and (pointer: coarse) {
		.expr-host {
			font-size: 16px;
		}
	}
</style>
