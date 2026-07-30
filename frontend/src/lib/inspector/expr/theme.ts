/**
 * The editor's look, bound to the token ladder (D-X2).
 *
 * Two families, and both are CSS-in-JS for the same unavoidable reason: CodeMirror GENERATES the class
 * names these rules paint (`.cm-content`, `.cm-tooltip-autocomplete`, the highlight classes), and the
 * completion popup is mounted in `document.body` rather than inside the component, so a Svelte-scoped
 * rule could not reach it. A theme extension is also the only form guaranteed to out-specify
 * CodeMirror's own base theme.
 *
 * What that must NOT mean is a second palette. Every value here is a `var(--…)` — the colours are the
 * conventional `--syn-*` family app.css mints, the geometry is the `--space-*`/`--radius-*`/`--hit`
 * rungs — and `theme.test.ts` walks these objects to prove it, so "centralized in location" is a test
 * rather than a claim. The spec object is exported for exactly that reason.
 */
import { HighlightStyle } from '@codemirror/language';
import { EditorView } from '@codemirror/view';
import { tags as t } from '@lezer/highlight';

/**
 * Tag → ink. The ROLE assignment is the conventional one every reader of a dark editor already knows
 * (it is One Dark's, which is what `@codemirror/theme-one-dark` ships); only the values are ours, and
 * they live in app.css.
 */
export const EXPR_HIGHLIGHT_SPEC = [
	{ tag: t.keyword, color: 'var(--syn-keyword)' },
	{ tag: [t.name, t.deleted, t.character, t.propertyName, t.macroName], color: 'var(--syn-name)' },
	{ tag: [t.function(t.variableName), t.labelName], color: 'var(--syn-function)' },
	{ tag: [t.atom, t.bool, t.special(t.variableName), t.constant(t.name)], color: 'var(--syn-literal)' },
	{ tag: [t.definition(t.name), t.separator], color: 'var(--syn-punct)' },
	{
		tag: [t.typeName, t.className, t.number, t.changed, t.annotation, t.modifier, t.self, t.namespace],
		color: 'var(--syn-type)'
	},
	{ tag: [t.operator, t.operatorKeyword, t.escape, t.regexp, t.special(t.string)], color: 'var(--syn-operator)' },
	{ tag: [t.meta, t.comment], color: 'var(--syn-comment)', fontStyle: 'italic' },
	{ tag: [t.processingInstruction, t.string, t.inserted], color: 'var(--syn-string)' },
	{ tag: t.invalid, color: 'var(--danger)' }
];

export const exprHighlight = HighlightStyle.define(EXPR_HIGHLIGHT_SPEC);

/** The editor chrome, the completion popup and the lint marks. */
export const EXPR_THEME_SPEC = {
	'&': {
		background: 'transparent',
		color: 'var(--text)',
		// The host element wears the app-wide focus ring; the base theme's dotted outline would double it.
		'&.cm-focused': { outline: 'none' }
	},
	'.cm-scroller': { fontFamily: 'inherit', lineHeight: 'var(--lh-text)' },
	'.cm-content': { padding: 'var(--space-2) var(--space-4)', caretColor: 'var(--accent)' },
	'&.cm-focused .cm-cursor': { borderLeftColor: 'var(--accent)' },
	'.cm-selectionBackground, &.cm-focused .cm-selectionBackground, .cm-content ::selection': {
		background: 'var(--accent-fill)'
	},
	'.cm-placeholder': { color: 'var(--text-muted)' },

	/* The completion popup. It lives in `document.body` (an inspector panel clips its own overflow),
	   and CodeMirror copies the view's theme classes onto the container it creates there — which is
	   what lets these rules reach it at all. */
	'.cm-tooltip': {
		background: 'var(--surface-glass)',
		border: '1px solid var(--border)',
		borderRadius: 'var(--radius-sm)',
		boxShadow: 'var(--shadow-1)',
		color: 'var(--text)',
		zIndex: 'var(--z-menu)'
	},
	'.cm-tooltip.cm-tooltip-autocomplete > ul': {
		fontFamily: 'inherit',
		fontSize: 'var(--fs-small)',
		// A height, not a spacing rung — `styleDrift.test.ts` draws that line for the stylesheet and
		// `theme.test.ts` draws it here: how tall the list may get before it scrolls is this list's shape.
		maxHeight: '14rem',
		'& > li': {
			display: 'flex',
			alignItems: 'center',
			gap: 'var(--space-4)',
			padding: 'var(--space-2) var(--space-4)'
		},
		'& > li[aria-selected]': { background: 'var(--accent-fill)', color: 'var(--text)' }
	},
	'.cm-completionLabel': { flex: '1' },
	// The matched substring reads as the accent rather than as an underline — one signal, and the same
	// "this is what you typed" ink the rest of the app uses.
	'.cm-completionMatchedText': { textDecoration: 'none', color: 'var(--accent)' },
	'.cm-completionDetail': { fontStyle: 'normal', color: 'var(--text-muted)' },
	'.cm-completionIcon': { color: 'var(--text-dim)', opacity: '1' },

	/* The diagnostic. `underline wavy` instead of the base theme's data-URI squiggle, whose colour is
	   baked into the SVG and could therefore only be restated as a literal. */
	'.cm-lintRange-error': {
		backgroundImage: 'none',
		textDecoration: 'underline wavy var(--danger)',
		textDecorationSkipInk: 'none'
	},
	'.cm-tooltip-lint': { fontFamily: 'inherit', fontSize: 'var(--fs-small)' },
	'.cm-diagnostic': { padding: 'var(--space-2) var(--space-4)', borderLeftColor: 'var(--danger)' },
	'.cm-diagnostic-error': { borderLeftColor: 'var(--danger)' },

	/* D-X8: the popup rows are tap targets like anything else. One coarse idiom, the same spelling the
	   rest of the app is pinned to; `16px` is the iOS focus-zoom threshold, an absolute device fact
	   rather than a scale rung, and it is the popup's own readability floor here. */
	'@media (hover: none) and (pointer: coarse)': {
		'.cm-tooltip.cm-tooltip-autocomplete > ul': { fontSize: '16px' },
		'.cm-tooltip.cm-tooltip-autocomplete > ul > li': { minHeight: 'var(--hit)' }
	}
};

export const exprTheme = EditorView.theme(EXPR_THEME_SPEC);
