<!-- Combobox — a text field over a closed list of names: a press opens the list, typing filters
     it, and only a name ON the list commits, by pick, Enter or blur. Typed text that is no name is
     dropped, so a half-typed name is never sent. -->
<script lang="ts">
	import type { HTMLAttributes } from 'svelte/elements';
	import { Icon } from 'panelty';
	import Popover from './Popover.svelte';
	import { claimFieldControlId } from './field';

	export interface ComboOption {
		label: string;
		detail?: string;
	}

	let {
		value,
		options,
		onCommit,
		placeholder = '',
		disabled = false,
		testid,
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		value: string;
		/** The names offered, read at the moment the list opens. */
		options: () => ComboOption[];
		onCommit: (label: string) => void;
		placeholder?: string;
		disabled?: boolean;
		/** Lands on the input; the list is `${testid}-list`. */
		testid: string;
	} = $props();

	const ownId = $props.id();
	const fieldId = claimFieldControlId(ownId);
	let box = $state<HTMLDivElement | null>(null);
	let input = $state<HTMLInputElement | null>(null);
	let open = $state(false);
	let typed = $state('');
	let all = $state<ComboOption[]>([]);
	let active = $state(0);
	$effect(() => {
		typed = value;
	});
	const shown = $derived.by(() => {
		const q = typed.trim().toLowerCase();
		return q === '' || q === value.toLowerCase() ? all : all.filter((o) => o.label.toLowerCase().includes(q));
	});
	$effect(() => {
		if (active >= shown.length) active = 0;
	});

	function show(): void {
		if (disabled) return;
		all = options();
		active = Math.max(0, all.findIndex((o) => o.label === value));
		open = true;
	}
	function close(): void {
		open = false;
		typed = value;
	}
	function commit(label: string): void {
		open = false;
		typed = label;
		if (label !== value) onCommit(label);
	}
	function settle(): void {
		const hit = all.find((o) => o.label === typed.trim());
		if (hit) commit(hit.label);
		else close();
	}
	function key(e: KeyboardEvent): void {
		if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
			e.preventDefault();
			if (!open) return show();
			const n = shown.length;
			if (n) active = (active + (e.key === 'ArrowDown' ? 1 : n - 1)) % n;
		} else if (e.key === 'Enter') {
			e.preventDefault();
			const pick = open ? shown[active] : undefined;
			if (pick) commit(pick.label);
			else settle();
		} else if (e.key === 'Escape' && open) {
			e.stopPropagation();
			close();
		}
	}
</script>

<div {...rest} bind:this={box} class={`ui-combo ${klass}`.trim()}>
	<input
		bind:this={input}
		id={fieldId}
		type="text"
		role="combobox"
		autocomplete="off"
		autocapitalize="off"
		spellcheck="false"
		aria-expanded={open}
		aria-controls={`${ownId}-list`}
		aria-autocomplete="list"
		class="ui-combo-input"
		{placeholder}
		{disabled}
		value={typed}
		data-testid={testid}
		onfocus={show}
		onpointerdown={() => {
			if (!open && document.activeElement === input) show();
		}}
		onblur={() => {
			if (open) settle();
		}}
		onkeydown={key}
		oninput={(e) => {
			typed = (e.currentTarget as HTMLInputElement).value;
			if (!open) show();
			active = 0;
		}}
	/>
	<span class="ui-combo-caret" aria-hidden="true"><Icon name="chevron-down" /></span>
</div>
{#if open}
	<Popover anchor={box} open onDismiss={close} class="ui-combo-list" data-testid={`${testid}-list`}>
		<div role="listbox" id={`${ownId}-list`} aria-label={placeholder || 'options'}>
			{#each shown as o, i (o.label)}
				<!-- svelte-ignore a11y_click_events_have_key_events -->
				<!-- svelte-ignore a11y_interactive_supports_focus -->
				<div
					class="ui-combo-option"
					class:active={i === active}
					role="option"
					aria-selected={o.label === value}
					onpointerdown={(e) => e.preventDefault()}
					onpointerenter={() => (active = i)}
					onclick={() => commit(o.label)}
				>
					<span class="ui-combo-name">{o.label}</span>
					{#if o.detail}<span class="ui-combo-detail">{o.detail}</span>{/if}
				</div>
			{:else}
				<div class="ui-combo-none">No match</div>
			{/each}
		</div>
	</Popover>
{/if}

<style>
	.ui-combo {
		position: relative;
		display: flex;
		align-items: stretch;
		flex: 1 1 auto;
		min-width: 0;
	}
	.ui-combo-input {
		flex: 1 1 auto;
		min-width: 0;
		width: 100%;
		padding-right: var(--space-6);
		color: var(--text);
		font-family: var(--font-mono);
	}
	.ui-combo-input:disabled {
		opacity: var(--disabled-opacity);
		cursor: not-allowed;
	}
	/* Decoration over the input's own right edge; the input owns every press. */
	.ui-combo-caret {
		position: absolute;
		top: 0;
		right: var(--space-1);
		bottom: 0;
		display: inline-flex;
		align-items: center;
		pointer-events: none;
		color: var(--text-muted);
	}
	.ui-combo-input:disabled + .ui-combo-caret {
		opacity: var(--disabled-opacity);
	}
	:global(.ui-combo-list) {
		max-height: 40vh;
		overflow-y: auto;
		padding: var(--space-1);
	}
	.ui-combo-option {
		display: flex;
		align-items: baseline;
		gap: var(--space-3);
		min-height: var(--hit);
		padding: var(--space-1) var(--space-3);
		border-radius: var(--radius-sm);
		font-family: var(--font-mono);
		font-size: var(--fs-small);
		cursor: pointer;
	}
	.ui-combo-option.active {
		background: var(--hover-fill);
	}
	.ui-combo-option[aria-selected='true'] .ui-combo-name {
		color: var(--accent);
	}
	.ui-combo-detail,
	.ui-combo-none {
		color: var(--text-muted);
		font-size: var(--fs-micro);
	}
	.ui-combo-none {
		padding: var(--space-2) var(--space-3);
	}
</style>
