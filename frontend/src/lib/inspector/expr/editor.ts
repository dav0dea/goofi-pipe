/** The mounted expression editor and the reference picker — one lazy CodeMirror chunk, two
 *  configurations of it: a Python expression with goofi's completions, and a bare picker whose only
 *  completions are the names it is handed. */
import { EditorState, Prec, type Extension } from '@codemirror/state';
import { EditorView, keymap, placeholder, tooltips, type KeyBinding } from '@codemirror/view';
import { syntaxHighlighting } from '@codemirror/language';
import { python } from '@codemirror/lang-python';
import {
	acceptCompletion,
	autocompletion,
	closeCompletion,
	startCompletion,
	type CompletionSource
} from '@codemirror/autocomplete';
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
import { setDiagnostics } from '@codemirror/lint';
import { MARGIN, overlayViewport } from 'panelty';
import { goofiLanguageData } from './complete';
import { expressionDiagnostics } from './diagnostics';
import { singleLineExpression } from './singleLine';
import { exprHighlight, exprTheme } from './theme';
import type { ExprCatalogue } from './catalogue';
import type { PickerOption } from './refs';

export interface ExprEditorOptions {
	doc: string;
	/** Read at the moment a completion is asked for. */
	catalogue: () => ExprCatalogue;
	/** Enter, or blur with a changed document. */
	onCommit: (value: string) => void;
	/** The backend's last compile/eval failure for this param, or null. */
	error: string | null;
	placeholder?: string;
	attributes: Record<string, string>;
}

export interface PickerOptions {
	doc: string;
	/** Read at the moment a completion is asked for; what is typed filters it. */
	options: () => PickerOption[];
	/** A pick from the list, Enter, or blur with a changed document. */
	onCommit: (value: string) => void;
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

/* Parented to `document.body` because an inspector panel clips its overflow, and sized against
   `overlayViewport()` — CodeMirror's default `innerHeight` would park the list under the keyboard. */
const popup = (): Extension =>
	tooltips({
		parent: document.body,
		position: 'fixed',
		tooltipSpace: () => {
			const vp = overlayViewport();
			return { top: MARGIN, left: MARGIN, bottom: vp.height - MARGIN, right: vp.width - MARGIN };
		}
	});

/** The commit-on-change discipline both editors share: one committed value, compared before each
 *  send, and adopted when the outside world moves it. */
interface Committer {
	send(view: EditorView): void;
	adopt(next: string): void;
}

function committer(doc: string, onCommit: (value: string) => void): Committer {
	let committed = doc;
	return {
		send: (view) => {
			const next = view.state.doc.toString();
			if (next === committed) return;
			committed = next;
			onCommit(next);
		},
		adopt: (next) => {
			committed = next;
		}
	};
}

function handleFor(
	view: EditorView,
	commit: (view: EditorView) => void,
	adopt: (next: string) => void,
	setError: (error: string | null) => void
): ExprEditorHandle {
	return {
		commit: () => commit(view),
		setValue: (next) => {
			if (next === view.state.doc.toString()) {
				adopt(next);
				return;
			}
			// A live echo must not yank the document from under live typing; the committed value is
			// left alone, so the local text still commits on blur.
			if (view.hasFocus) return;
			view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: next } });
			adopt(next);
		},
		setError,
		focus: () => {
			view.focus();
			view.dispatch({ selection: { anchor: view.state.doc.length } });
		},
		destroy: () => view.destroy()
	};
}

export function createExprEditor(host: HTMLElement, opts: ExprEditorOptions): ExprEditorHandle {
	const { send: commit, adopt } = committer(opts.doc, opts.onCommit);
	/* Escape must fall THROUGH once there is no popup: it is the app's, and it dismisses the auto
	   inspector pane. */
	const keys: KeyBinding[] = [
		{ key: 'Escape', run: (view) => closeCompletion(view) },
		{ key: 'Enter', run: (view) => acceptCompletion(view) || (commit(view), true) }
	];
	const extensions: Extension[] = [
		python(),
		goofiLanguageData(opts.catalogue),
		syntaxHighlighting(exprHighlight),
		autocompletion(),
		history(),
		exprTheme,
		popup(),
		EditorView.contentAttributes.of(opts.attributes),
		Prec.high(keymap.of(keys)),
		keymap.of([...historyKeymap, ...defaultKeymap]),
		EditorView.domEventHandlers({
			// The completion popup does not blur the editor, so accepting an option never races this.
			blur: (_e, view) => {
				commit(view);
				return false;
			}
		}),
		singleLineExpression
	];
	if (opts.placeholder) extensions.push(placeholder(opts.placeholder));
	const view = new EditorView({ state: EditorState.create({ doc: opts.doc, extensions }), parent: host });
	const showError = (error: string | null): void => {
		view.dispatch(setDiagnostics(view.state, expressionDiagnostics(error, view.state.doc)));
	};
	showError(opts.error);
	return handleFor(view, commit, adopt, showError);
}

/** A field whose only legal contents are the names it is handed: no language, no highlighting, the
 *  list opens on focus, what is typed filters it, and a pick commits at once. Typed text commits
 *  only when it IS one of the names — a half-typed name is never sent. */
export function createPicker(host: HTMLElement, opts: PickerOptions): ExprEditorHandle {
	const { send, adopt } = committer(opts.doc, opts.onCommit);
	const commit = (view: EditorView): void => {
		const typed = view.state.doc.toString();
		if (opts.options().some((o) => o.label === typed)) send(view);
	};
	const source: CompletionSource = (ctx) => {
		const typed = ctx.state.doc.toString().toLowerCase();
		const options = opts
			.options()
			.filter((o) => o.label.toLowerCase().includes(typed))
			.map((o) => ({
				label: o.label,
				detail: o.detail,
				type: 'variable',
				apply: (view: EditorView) => {
					view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: o.label } });
					send(view);
				}
			}));
		return options.length ? { from: 0, to: ctx.state.doc.length, options, filter: false } : null;
	};
	const keys: KeyBinding[] = [
		{ key: 'Escape', run: (view) => closeCompletion(view) },
		{ key: 'Enter', run: (view) => acceptCompletion(view) || (commit(view), true) }
	];
	const extensions: Extension[] = [
		autocompletion({ override: [source], activateOnTyping: true }),
		history(),
		exprTheme,
		popup(),
		EditorView.contentAttributes.of(opts.attributes),
		Prec.high(keymap.of(keys)),
		keymap.of([...historyKeymap, ...defaultKeymap]),
		EditorView.domEventHandlers({
			focus: (_e, view) => {
				startCompletion(view);
				return false;
			},
			blur: (_e, view) => {
				commit(view);
				return false;
			}
		}),
		singleLineExpression
	];
	if (opts.placeholder) extensions.push(placeholder(opts.placeholder));
	const view = new EditorView({ state: EditorState.create({ doc: opts.doc, extensions }), parent: host });
	return handleFor(view, commit, adopt, () => {});
}
