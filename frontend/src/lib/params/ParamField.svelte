<script lang="ts">
	import type { ParamDescriptor } from '$lib/api/types';

	type Props = {
		paramName: string;
		descriptor: ParamDescriptor;
		onCommit: (value: unknown) => void;
	};
	const { paramName, descriptor, onCommit }: Props = $props();

	let local = $state<unknown>(descriptor.value);
	// Suppress backend echoes while the user is actively editing — without
	// this, every committed slider/text edit would re-set `local` to the
	// echoed value mid-typing.
	let editing = $state(false);

	$effect(() => {
		if (!editing) local = descriptor.value;
	});

	function commit(v: unknown): void {
		local = v;
		onCommit(v);
	}

	// Auto-extend slider bounds when the live value lies outside [vmin, vmax].
	// goofi3 does the same — without it, a value of 5.0 on a [0, 1] slider
	// would render at the right edge and clip on edit.
	const numeric = $derived(
		descriptor.type === 'float' || descriptor.type === 'int' ? descriptor : null
	);
	const lo = $derived(
		numeric ? Math.min(numeric.vmin, Number(local ?? 0)) : 0
	);
	const hi = $derived(
		numeric ? Math.max(numeric.vmax, Number(local ?? 0)) : 1
	);
	const step = $derived(
		numeric ? (numeric.type === 'int' ? 1 : Math.max((hi - lo) / 200, 1e-6)) : 1
	);
</script>

<div class="field" title={descriptor.doc ?? ''}>
	<div class="label-row">
		<span class="label">{paramName}</span>
	</div>
	<div class="control">
		{#if descriptor.type === 'bool'}
			{#if descriptor.trigger}
				<button class="trigger" onclick={() => commit(true)}>{paramName}</button>
			{:else}
				<label class="switch">
					<input
						type="checkbox"
						checked={Boolean(local)}
						onchange={(e) => commit((e.currentTarget as HTMLInputElement).checked)}
					/>
					<span class="slider"></span>
				</label>
				<span class="bool-label">{local ? 'on' : 'off'}</span>
			{/if}
		{:else if descriptor.type === 'float' || descriptor.type === 'int'}
			<input
				type="range"
				min={lo}
				max={hi}
				{step}
				value={Number(local)}
				onpointerdown={() => (editing = true)}
				onpointerup={() => {
					editing = false;
					commit(Number(local));
				}}
				oninput={(e) => {
					local = Number((e.currentTarget as HTMLInputElement).value);
				}}
				onchange={(e) => commit(Number((e.currentTarget as HTMLInputElement).value))}
			/>
			<input
				class="num"
				type="number"
				step={descriptor.type === 'int' ? 1 : 'any'}
				value={Number(local)}
				onfocus={() => (editing = true)}
				onblur={() => {
					editing = false;
					commit(Number(local));
				}}
				onkeydown={(e) => {
					if (e.key === 'Enter') (e.currentTarget as HTMLInputElement).blur();
				}}
				oninput={(e) => {
					local = Number((e.currentTarget as HTMLInputElement).value);
				}}
			/>
		{:else if descriptor.type === 'string'}
			{#if descriptor.options && descriptor.options.length > 0}
				<select
					value={String(local ?? '')}
					onchange={(e) => commit((e.currentTarget as HTMLSelectElement).value)}
				>
					{#each descriptor.options as opt}
						<option value={opt}>{opt}</option>
					{/each}
				</select>
			{:else}
				<input
					type="text"
					value={String(local ?? '')}
					onfocus={() => (editing = true)}
					onblur={(e) => {
						editing = false;
						commit((e.currentTarget as HTMLInputElement).value);
					}}
					onkeydown={(e) => {
						if (e.key === 'Enter') (e.currentTarget as HTMLInputElement).blur();
					}}
					oninput={(e) => {
						local = (e.currentTarget as HTMLInputElement).value;
					}}
				/>
			{/if}
		{:else}
			<code>{JSON.stringify(local)}</code>
		{/if}
	</div>
</div>

<style>
	.field {
		display: flex;
		flex-direction: column;
		gap: 3px;
		font-size: 11px;
	}
	.label-row {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.label {
		color: var(--text-dim);
		font-family: var(--font-mono);
		font-size: 10px;
		text-transform: lowercase;
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		letter-spacing: 0.02em;
	}
	.control {
		display: flex;
		align-items: center;
		gap: 6px;
		min-width: 0;
	}
	input[type='range'] {
		flex-grow: 1;
		min-width: 0;
		accent-color: var(--accent);
		background: transparent;
		padding: 0;
		border: none;
		height: 18px;
	}
	.num {
		width: 72px;
		text-align: right;
		font-variant-numeric: tabular-nums;
	}
	input[type='text'],
	select {
		flex-grow: 1;
		min-width: 0;
	}
	.trigger {
		width: 100%;
		background: color-mix(in srgb, var(--accent) 22%, transparent);
		border-color: var(--accent);
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.trigger:hover:not(:disabled) {
		background: var(--accent);
		color: #0a0c10;
	}
	.switch {
		position: relative;
		width: 34px;
		height: 18px;
		display: inline-block;
		flex-shrink: 0;
	}
	.switch input {
		opacity: 0;
		position: absolute;
		inset: 0;
		cursor: pointer;
	}
	.slider {
		position: absolute;
		inset: 0;
		background: var(--bg-elev-3);
		border: 1px solid var(--border);
		border-radius: 999px;
		transition: background 80ms ease;
	}
	.slider::before {
		content: '';
		position: absolute;
		left: 2px;
		top: 1px;
		width: 12px;
		height: 12px;
		border-radius: 50%;
		background: var(--text-dim);
		transition: transform 120ms ease;
	}
	.switch input:checked + .slider {
		background: color-mix(in srgb, var(--accent) 35%, transparent);
		border-color: var(--accent);
	}
	.switch input:checked + .slider::before {
		transform: translateX(15px);
		background: var(--accent);
	}
	.bool-label {
		color: var(--text-faint);
		font-family: var(--font-mono);
		font-size: 10px;
		letter-spacing: 0.02em;
	}
	code {
		font-family: var(--font-mono);
		font-size: 10px;
		color: var(--text-faint);
	}
</style>
