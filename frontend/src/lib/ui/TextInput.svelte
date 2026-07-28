<!--
  TextInput — a dumb text control (spec §2.2, §4): `value` in, `onChange` out, committing on blur /
  Enter via the shared `useLiveValue` latch (echoes suppressed while typing). Its one variant axis is
  `inputmode` (`text | decimal | search | path`), which sets the right mobile keyboard plus the
  matching `enterkeyhint` / `autocapitalize` / `autocorrect` / `spellcheck` — collapsing the audit's
  zero-`inputmode` gap into one closed union (F's responsive-keyboard requirement, baked in here).

  `path` maps to the `url` inputmode: that virtual keyboard surfaces `/` and `.` and drops the space
  bar — genuinely the filesystem-path keyboard — and keeps every variant's inputmode distinct. The
  input is the root: `class` merged, `data-testid` (and any other attribute) forwarded via `...rest`;
  it claims the enclosing Field's label id so clicking the label focuses it.
-->
<script lang="ts">
	import type { HTMLInputAttributes } from 'svelte/elements';
	import { useLiveValue } from './liveValue.svelte';
	import { claimFieldControlId } from './field';

	type InputModeVariant = 'text' | 'decimal' | 'search' | 'path';

	// One source of truth for the per-variant keyboard + editing hints. `autocorrect` is Safari's
	// non-standard attribute; carried as a plain attribute bag so it applies without a typed slot.
	const MODE_ATTRS: Record<InputModeVariant, Record<string, string>> = {
		text: { inputmode: 'text', enterkeyhint: 'done', autocapitalize: 'sentences', autocorrect: 'on', spellcheck: 'true' },
		decimal: { inputmode: 'decimal', enterkeyhint: 'done', autocapitalize: 'off', autocorrect: 'off', spellcheck: 'false' },
		search: { inputmode: 'search', enterkeyhint: 'search', autocapitalize: 'off', autocorrect: 'off', spellcheck: 'false' },
		path: { inputmode: 'url', enterkeyhint: 'go', autocapitalize: 'off', autocorrect: 'off', spellcheck: 'false' }
	};

	let {
		value,
		onChange,
		inputmode = 'text',
		class: klass = '',
		...rest
	}: Omit<HTMLInputAttributes, 'value' | 'type' | 'inputmode' | 'oninput' | 'onchange'> & {
		value: string;
		onChange: (v: string) => void;
		inputmode?: InputModeVariant;
	} = $props();

	const ownId = $props.id();
	const fieldId = claimFieldControlId(ownId);
	const live = useLiveValue<string>(
		() => value,
		(v) => onChange(v)
	);
	const modeAttrs = $derived(MODE_ATTRS[inputmode]);
</script>

<input
	{...rest}
	{...modeAttrs}
	id={fieldId}
	type="text"
	class={`ui-text ${klass}`.trim()}
	value={live.value}
	onfocus={() => live.begin()}
	onblur={() => {
		live.commit(live.value);
		live.end();
	}}
	onkeydown={(e) => {
		if (e.key === 'Enter') (e.currentTarget as HTMLInputElement).blur();
	}}
	oninput={(e) => live.input((e.currentTarget as HTMLInputElement).value)}
/>

<style>
	/* Inherits the app-wide input chrome + coarse --hit floor; just fill its container. */
	.ui-text {
		flex: 1 1 auto;
		min-width: 0;
		width: 100%;
		color: var(--text);
	}
	.ui-text:disabled {
		opacity: var(--disabled-opacity);
		cursor: not-allowed;
	}
</style>
