<script lang="ts">
	import type { ParamDescriptor } from '$lib/api/types';

	type Props = {
		paramName: string;
		descriptor: ParamDescriptor;
		onCommit: (value: unknown) => void;
	};
	const { paramName, descriptor, onCommit }: Props = $props();

	let local = $state<unknown>(descriptor.value);
	let dragging = $state(false);

	$effect(() => {
		// Sync from backend updates unless the user is currently editing.
		if (!dragging) local = descriptor.value;
	});

	function commit(v: unknown): void {
		local = v;
		onCommit(v);
	}
</script>

<div class="field" title={descriptor.doc ?? ''}>
	<label>{paramName}</label>
	<div class="control">
		{#if descriptor.type === 'bool'}
			{#if descriptor.trigger}
				<button class="trigger" onclick={() => commit(true)}>trigger</button>
			{:else}
				<label class="switch">
					<input
						type="checkbox"
						checked={Boolean(local)}
						onchange={(e) => commit((e.currentTarget as HTMLInputElement).checked)}
					/>
					<span class="slider"></span>
				</label>
			{/if}
		{:else if descriptor.type === 'float' || descriptor.type === 'int'}
			<input
				type="range"
				min={descriptor.vmin}
				max={descriptor.vmax}
				step={descriptor.type === 'int' ? 1 : (descriptor.vmax - descriptor.vmin) / 200 || 0.01}
				value={Number(local)}
				onpointerdown={() => (dragging = true)}
				onpointerup={() => {
					dragging = false;
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
				onfocus={() => (dragging = true)}
				onblur={() => {
					dragging = false;
					commit(Number(local));
				}}
				onchange={(e) => commit(Number((e.currentTarget as HTMLInputElement).value))}
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
					onfocus={() => (dragging = true)}
					onblur={(e) => {
						dragging = false;
						commit((e.currentTarget as HTMLInputElement).value);
					}}
					onkeydown={(e) => {
						if (e.key === 'Enter') {
							(e.currentTarget as HTMLInputElement).blur();
						}
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
		display: grid;
		grid-template-columns: 90px 1fr;
		align-items: center;
		gap: 8px;
		font-size: 11px;
	}
	label {
		color: var(--text-dim);
		font-family: var(--font-mono);
		font-size: 10px;
		text-transform: lowercase;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
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
	}
	.num {
		width: 72px;
	}
	input[type='text'],
	select {
		flex-grow: 1;
		min-width: 0;
	}
	.trigger {
		width: 100%;
		background: color-mix(in srgb, var(--accent) 25%, transparent);
		border-color: var(--accent);
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
	code {
		font-family: var(--font-mono);
		font-size: 10px;
		color: var(--text-faint);
	}
</style>
