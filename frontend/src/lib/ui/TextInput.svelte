<!-- TextInput — a dumb text control: `value` in, `onChange` out, committed on blur / Enter. Its
     one variant axis is `inputmode`, which carries the keyboard and editing hints. -->
<script lang="ts">
	import type { HTMLInputAttributes } from 'svelte/elements';
	import { useLiveValue } from './liveValue.svelte';
	import { claimFieldControlId } from './field';
	import { MODE_ATTRS, type InputModeVariant } from './inputMode';

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
