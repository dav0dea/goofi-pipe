/** Per-variant virtual-keyboard and editing hints. `search` is the IDENTIFIER variant — a node,
 * tab or global name is machine-read — and `path` maps to the `url` keyboard, which carries `/`. */

export type InputModeVariant = 'text' | 'decimal' | 'search' | 'path';

/** A plain attribute bag: `autocorrect` is Safari's non-standard attribute and has no typed slot. */
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
