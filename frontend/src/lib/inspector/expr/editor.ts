/**
 * The mounted expression editor — and the ROOT OF THE LAZY CHUNK (D-X5). Everything CodeMirror is
 * reachable only from here, so `load.ts`'s `await import()` is the one edge that pulls it in and the
 * main bundle stays exactly as heavy as it was before X. `ExprEditor.svelte` never imports a
 * `@codemirror/*` module itself; it holds a handle to what this returns.
 *
 * ONE editor in two modes (D-X1), not two implementations. The mode changes three things and nothing
 * else: whether the document may hold more than one line, what Enter does, and what Escape does.
 *
 * Commit is COARSE (D-X6) — each one is a `set_param_expr` RPC plus a re-eval, so it fires on
 * Enter/blur (inline) or ⌘/Ctrl+Enter (expanded), never per keystroke. Highlighting, completion and
 * the diagnostics are live; only the commit is not.
 */
import { EditorState, Prec, type Extension } from '@codemirror/state';
import { EditorView, keymap, placeholder, tooltips, type KeyBinding } from '@codemirror/view';
import { syntaxHighlighting } from '@codemirror/language';
import { python } from '@codemirror/lang-python';
import { acceptCompletion, autocompletion, closeCompletion } from '@codemirror/autocomplete';
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
import { setDiagnostics } from '@codemirror/lint';
import { overlayViewport } from '$lib/ui';
import { goofiLanguageData } from './complete';
import { expressionDiagnostics } from './diagnostics';
import { singleLineExpression } from './singleLine';
import { exprHighlight, exprTheme } from './theme';
import type { ExprCatalogue } from './catalogue';

/** The margin the popup keeps off every viewport edge — `clampToViewport`'s, so the two agree. */
const POPUP_MARGIN = 6;

export interface ExprEditorOptions {
	/** The initial source. */
	doc: string;
	/** Expanded mode: newlines allowed, ⌘/Ctrl+Enter commits, Escape cancels. */
	multiline: boolean;
	/** Read at the moment a completion is asked for, so a node added while the editor is open completes. */
	catalogue: () => ExprCatalogue;
	/** Enter (inline) / ⌘-Enter (expanded) / blur with a changed document. */
	onCommit: (value: string) => void;
	/** Escape in expanded mode, with no completion popup to close first. */
	onCancel: () => void;
	/** The backend's last compile/eval failure for this param, or null. */
	error: string | null;
	placeholder?: string;
	/** Attributes for the editable element — `data-testid`, `aria-label`. */
	attributes: Record<string, string>;
}

export interface ExprEditorHandle {
	/** The live document. */
	value(): string;
	/** Commit now, if the document has changed — what the expanded editor's `apply` control asks for.
	 *  A code editor's document is not a bindable string, so the owner asks for the commit rather than
	 *  mirroring every keystroke into its own state. */
	commit(): void;
	/** Adopt an externally changed source. A no-op while the user is typing into it. */
	setValue(next: string): void;
	setError(error: string | null): void;
	focus(): void;
	destroy(): void;
}

export function createExprEditor(host: HTMLElement, opts: ExprEditorOptions): ExprEditorHandle {
	/** The last value this editor and the backend agree on — the latch that keeps a commit from
	 *  firing when nothing changed, and every commit from being one RPC per keystroke. */
	let committed = opts.doc;

	const commit = (view: EditorView): void => {
		const next = view.state.doc.toString();
		if (next === committed) return;
		committed = next;
		opts.onCommit(next);
	};

	/* Escape is LAYERED (D-X7): close the completion popup, else — expanded only — cancel. The inline
	   field deliberately falls THROUGH once there is no popup, because Escape there is the app's (it
	   dismisses the auto inspector pane), and swallowing it would take an affordance away.
	   `completionKeymap` sits at highest precedence and already closes an open popup, so the
	   `closeCompletion` call is the belt to that braces rather than the mechanism. */
	const keys: KeyBinding[] = [
		{
			key: 'Escape',
			run: (view) => {
				if (closeCompletion(view)) return true;
				if (!opts.multiline) return false;
				opts.onCancel();
				return true;
			}
		}
	];
	if (opts.multiline) {
		// Enter keeps its newline; the chord commits, exactly as the textarea did.
		keys.push({ key: 'Mod-Enter', run: (view) => (commit(view), true) });
	} else {
		// Single line: Enter is the commit — but an open popup owns it first, and saying so here means
		// it stays true if precedence ever moves.
		keys.push({ key: 'Enter', run: (view) => acceptCompletion(view) || (commit(view), true) });
	}

	const extensions: Extension[] = [
		python(),
		// THE seam: our completions registered as Python language data, so CodeMirror ranks them into
		// the same popup as its own sources instead of a second one.
		goofiLanguageData(opts.catalogue),
		syntaxHighlighting(exprHighlight),
		autocompletion(),
		history(),
		exprTheme,
		/* The popup is parented to `document.body` — an inspector panel clips its own overflow, so an
		   in-editor tooltip would be cut off — and constrained by `overlayViewport()`, the app's one
		   soft-keyboard-aware measurement (D-X8). `tooltipSpace` is the designed hook for that;
		   CodeMirror's own default reads `innerHeight`, which a keyboard does not shrink, and would
		   park the list underneath the very keyboard that raised it. */
		tooltips({
			parent: document.body,
			position: 'fixed',
			tooltipSpace: () => {
				const vp = overlayViewport();
				return {
					top: POPUP_MARGIN,
					left: POPUP_MARGIN,
					bottom: vp.height - POPUP_MARGIN,
					right: vp.width - POPUP_MARGIN
				};
			}
		}),
		EditorView.contentAttributes.of(opts.attributes),
		Prec.high(keymap.of(keys)),
		keymap.of([...historyKeymap, ...defaultKeymap]),
		EditorView.domEventHandlers({
			// Inline commits on blur, matching what the plain `TextInput` did. The completion popup does
			// not blur the editor (CodeMirror preventDefaults the mousedown on its list), so accepting an
			// option never races this.
			blur: (_e, view) => {
				if (!opts.multiline) commit(view);
				return false;
			}
		})
	];
	if (opts.placeholder) extensions.push(placeholder(opts.placeholder));
	if (!opts.multiline) extensions.push(singleLineExpression);

	const view = new EditorView({
		state: EditorState.create({ doc: opts.doc, extensions }),
		parent: host
	});
	const showError = (error: string | null): void => {
		view.dispatch(setDiagnostics(view.state, expressionDiagnostics(error, view.state.doc)));
	};
	showError(opts.error);

	return {
		value: () => view.state.doc.toString(),
		commit: () => commit(view),
		setValue: (next) => {
			if (next === view.state.doc.toString()) {
				committed = next;
				return;
			}
			// A live backend echo must not yank the document out from under someone typing into it —
			// the same latch `$lib/ui`'s `useLiveValue` gives every other control. `committed` is left
			// alone, so the local text still commits on blur.
			if (view.hasFocus) return;
			view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: next } });
			committed = next;
		},
		setError: showError,
		focus: () => {
			view.focus();
			view.dispatch({ selection: { anchor: view.state.doc.length } });
		},
		destroy: () => view.destroy()
	};
}
