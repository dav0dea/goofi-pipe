<!-- TextArea — TextInput's multi-line twin: `value` in, `onChange` out, committed on blur. Enter
     is a NEWLINE here and never a commit, which is the whole reason this is not TextInput with a
     taller box: a field people write a poem into cannot spend its Return key on submitting. -->
<script lang="ts">
	import type { HTMLTextareaAttributes } from 'svelte/elements';
	import { useLiveValue } from './liveValue.svelte';
	import { claimFieldControlId } from './field';

	let {
		value,
		onChange,
		class: klass = '',
		...rest
	}: Omit<HTMLTextareaAttributes, 'value' | 'oninput' | 'onchange'> & {
		value: string;
		onChange: (v: string) => void;
	} = $props();

	const ownId = $props.id();
	const fieldId = claimFieldControlId(ownId);
	const live = useLiveValue<string>(
		() => value,
		(v) => onChange(v)
	);
</script>

<textarea
	{...rest}
	id={fieldId}
	class={`ui-textarea ${klass}`.trim()}
	spellcheck="false"
	autocomplete="off"
	value={live.value}
	onfocus={() => live.begin()}
	onblur={() => {
		live.commit(live.value);
		live.end();
	}}
	onkeydown={(e) => {
		// Escape gives the board its keys back without committing a half-written line; every other
		// key, Enter included, belongs to the text.
		if (e.key === 'Escape') (e.currentTarget as HTMLTextAreaElement).blur();
		else e.stopPropagation();
	}}
	oninput={(e) => live.input((e.currentTarget as HTMLTextAreaElement).value)}
></textarea>

<style>
	.ui-textarea {
		flex: 1 1 auto;
		width: 100%;
		height: 100%;
		min-width: 0;
		min-height: 0;
		resize: none;
		color: var(--text);
		font: inherit;
		line-height: 1.45;
		padding: var(--space-1) var(--space-2);
		overflow: auto;
	}
	.ui-textarea:disabled {
		opacity: var(--disabled-opacity);
		cursor: not-allowed;
	}
</style>
