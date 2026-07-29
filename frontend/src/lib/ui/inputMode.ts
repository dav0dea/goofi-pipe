/**
 * The per-variant virtual-keyboard + editing hints, in one place.
 *
 * `TextInput` is the primary consumer (its `inputmode` prop IS this union), but a handful of raw
 * `<input>`s legitimately stay native — the ones that need per-keystroke `bind:value`, an
 * Enter/Escape pair and `autofocus`, none of which a commit-on-blur primitive offers. They were
 * bypassing the union entirely and shipping the UA defaults: a phone keyboard with no `/`, sentence
 * capitalisation and a spellcheck squiggle on a node name. Spreading the same bag is what keeps
 * them in the app's one vocabulary instead of hand-rolling four attributes each.
 *
 * `search` is this codebase's IDENTIFIER variant, not merely its search box: what a node name, a
 * tab name, a global name and a global's value all need is autocapitalisation, autocorrection and
 * spellcheck OFF, because they are machine-read (an expression resolves `globals.<name>`), and
 * `search` is the variant that says so. `path` maps to the `url` inputmode: that keyboard surfaces
 * `/` and `.` and drops the space bar — genuinely the filesystem keyboard — which also keeps every
 * variant's inputmode distinct.
 */

export type InputModeVariant = 'text' | 'decimal' | 'search' | 'path';

/** `autocorrect` is Safari's non-standard attribute; carried as a plain attribute bag so it
 * applies without a typed slot. */
export const MODE_ATTRS: Record<InputModeVariant, Record<string, string>> = {
	text: {
		inputmode: 'text',
		enterkeyhint: 'done',
		autocapitalize: 'sentences',
		autocorrect: 'on',
		spellcheck: 'true'
	},
	decimal: {
		inputmode: 'decimal',
		enterkeyhint: 'done',
		autocapitalize: 'off',
		autocorrect: 'off',
		spellcheck: 'false'
	},
	search: {
		inputmode: 'search',
		enterkeyhint: 'search',
		autocapitalize: 'off',
		autocorrect: 'off',
		spellcheck: 'false'
	},
	path: {
		inputmode: 'url',
		enterkeyhint: 'go',
		autocapitalize: 'off',
		autocorrect: 'off',
		spellcheck: 'false'
	}
};
