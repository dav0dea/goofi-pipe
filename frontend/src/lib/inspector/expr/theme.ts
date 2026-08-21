/** The editor's look. CSS-in-JS because CodeMirror generates these class names and mounts the popup in
 *  `document.body`, out of reach of a Svelte-scoped rule; every value stays a `var(--…)`. */
import { HighlightStyle } from '@codemirror/language';
import { EditorView } from '@codemirror/view';
import { tags as t } from '@lezer/highlight';

/** Tag → ink. The role assignment is One Dark's; only the values are ours. */
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

export const EXPR_THEME_SPEC = {
	'&': {
		background: 'transparent',
		color: 'var(--text)',
		// The host element wears the app-wide focus ring; this would double it.
		'&.cm-focused': { outline: 'none' }
	},
	'.cm-scroller': { fontFamily: 'inherit', lineHeight: 'var(--lh-text)' },
	'.cm-content': { padding: 'var(--space-2) var(--space-4)', caretColor: 'var(--accent)' },
	'&.cm-focused .cm-cursor': { borderLeftColor: 'var(--accent)' },
	'.cm-selectionBackground, &.cm-focused .cm-selectionBackground, .cm-content ::selection': {
		background: 'var(--accent-fill)'
	},
	'.cm-placeholder': { color: 'var(--text-muted)' },

	/* The completion popup: CodeMirror copies the view's theme classes onto the container it creates
	   in `document.body`, which is what lets these rules reach it. */
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
	'.cm-completionMatchedText': { textDecoration: 'none', color: 'var(--accent)' },
	'.cm-completionDetail': { fontStyle: 'normal', color: 'var(--text-muted)' },
	'.cm-completionIcon': { color: 'var(--text-dim)', opacity: '1' },

	/* `underline wavy` rather than the base theme's data-URI squiggle, whose colour is baked into the SVG. */
	'.cm-lintRange-error': {
		backgroundImage: 'none',
		textDecoration: 'underline wavy var(--danger)',
		textDecorationSkipInk: 'none'
	},
	'.cm-tooltip-lint': { fontFamily: 'inherit', fontSize: 'var(--fs-small)' },
	'.cm-diagnostic': { padding: 'var(--space-2) var(--space-4)', borderLeftColor: 'var(--danger)' },
	'.cm-diagnostic-error': { borderLeftColor: 'var(--danger)' },

	/* `16px` is the iOS focus-zoom threshold, a device fact rather than a scale rung. */
	'@media (hover: none) and (pointer: coarse)': {
		'.cm-tooltip.cm-tooltip-autocomplete > ul': { fontSize: '16px' },
		'.cm-tooltip.cm-tooltip-autocomplete > ul > li': { minHeight: 'var(--hit)' }
	}
};

export const exprTheme = EditorView.theme(EXPR_THEME_SPEC);
