/** The mounted expression editor, and the root of the lazy CodeMirror chunk. One editor in two modes:
 *  the mode changes only whether the document may hold more than one line and what Enter/Escape do. */
import { EditorState, Prec, type Extension } from '@codemirror/state';
import { EditorView, keymap, placeholder, tooltips, type KeyBinding } from '@codemirror/view';
import { syntaxHighlighting } from '@codemirror/language';
import { python } from '@codemirror/lang-python';
import { acceptCompletion, autocompletion, closeCompletion } from '@codemirror/autocomplete';
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
import { setDiagnostics } from '@codemirror/lint';
import { MARGIN, overlayViewport } from 'panelty';
import { goofiLanguageData } from './complete';
import { expressionDiagnostics } from './diagnostics';
import { singleLineExpression } from './singleLine';
import { exprHighlight, exprTheme } from './theme';
import type { ExprCatalogue } from './catalogue';

export interface ExprEditorOptions {
	doc: string;
	/** Expanded mode: newlines allowed, ⌘/Ctrl+Enter commits, Escape cancels. */
	multiline: boolean;
	/** Read at the moment a completion is asked for. */
	catalogue: () => ExprCatalogue;
	/** Enter (inline) / ⌘-Enter (expanded) / blur with a changed document. */
	onCommit: (value: string) => void;
	/** Escape in expanded mode, with no completion popup to close first. */
	onCancel: () => void;
	/** The backend's last compile/eval failure for this param, or null. */
	error: string | null;
	placeholder?: string;
	attributes: Record<string, string>;
}

export interface ExprEditorHandle {
	/** Commit now, if the document has changed — a code editor's document is not a bindable string. */
	commit(): void;
	/** Adopt an externally changed source. A no-op while the user is typing into it. */
	setValue(next: string): void;
	setError(error: string | null): void;
	focus(): void;
	destroy(): void;
}

export function createExprEditor(host: HTMLElement, opts: ExprEditorOptions): ExprEditorHandle {
	/** The last value this editor and the backend agree on. */
	let committed = opts.doc;

	const commit = (view: EditorView): void => {
		const next = view.state.doc.toString();
		if (next === committed) return;
		committed = next;
		opts.onCommit(next);
	};

	/* Escape must fall THROUGH in the inline field once there is no popup: it is the app's there, and
	   it dismisses the auto inspector pane. */
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
		keys.push({ key: 'Mod-Enter', run: (view) => (commit(view), true) });
	} else {
		keys.push({ key: 'Enter', run: (view) => acceptCompletion(view) || (commit(view), true) });
	}

	const extensions: Extension[] = [
		python(),
		goofiLanguageData(opts.catalogue),
		syntaxHighlighting(exprHighlight),
		autocompletion(),
		history(),
		exprTheme,
		/* Parented to `document.body` because an inspector panel clips its overflow, and sized against
		   `overlayViewport()` — CodeMirror's default `innerHeight` would park the list under the keyboard. */
		tooltips({
			parent: document.body,
			position: 'fixed',
			tooltipSpace: () => {
				const vp = overlayViewport();
				return {
					top: MARGIN,
					left: MARGIN,
					bottom: vp.height - MARGIN,
					right: vp.width - MARGIN
				};
			}
		}),
		EditorView.contentAttributes.of(opts.attributes),
		Prec.high(keymap.of(keys)),
		keymap.of([...historyKeymap, ...defaultKeymap]),
		EditorView.domEventHandlers({
			// The completion popup does not blur the editor, so accepting an option never races this.
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
		commit: () => commit(view),
		setValue: (next) => {
			if (next === view.state.doc.toString()) {
				committed = next;
				return;
			}
			// A live echo must not yank the document from under live typing; `committed` is left alone, so
			// the local text still commits on blur.
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
