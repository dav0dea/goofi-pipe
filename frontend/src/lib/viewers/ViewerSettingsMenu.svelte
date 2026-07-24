<!--
  Per-slot viewer settings — a cog button that opens a Blender-style dropdown of
  collapsible setting groups, driven by the viewer's settings schema. Portaled to
  <body> so it escapes the zoomed/clipped SvelteFlow viewport.
-->
<script lang="ts">
	import { settingsSchemaFor, type SettingDescriptor, type SettingValue } from './settingsSchema';
	import type { ViewBinding } from './viewBinding';
	import { portal } from '$lib/workspace/portal';

	let { binding }: { binding: ViewBinding } = $props();

	const kind = $derived(binding.kind);
	const groups = $derived(settingsSchemaFor(kind));
	const settings = $derived(binding.settings);

	let open = $state(false);
	let anchor = $state<{ x: number; y: number }>({ x: 0, y: 0 });
	let collapsed = $state<Record<string, boolean>>({});

	const MENU_W = 212;

	function toggle(e: MouseEvent): void {
		e.stopPropagation();
		if (open) {
			open = false;
			return;
		}
		const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
		anchor = {
			x: Math.max(8, Math.min(r.left, window.innerWidth - MENU_W - 8)),
			y: Math.min(r.bottom + 4, window.innerHeight - 40)
		};
		open = true;
	}

	function visible(s: SettingDescriptor): boolean {
		return !s.showWhen || settings[s.showWhen.key] === s.showWhen.equals;
	}
	// Writes go straight to the binding, which owns persistence (inline →
	// node.viewers; panel → layout).
	function set(key: string, value: SettingValue): void {
		binding.setSetting(key, value);
	}
	function toggleGroup(title: string): void {
		collapsed = { ...collapsed, [title]: !collapsed[title] };
	}
</script>

<button class="cog" class:on={open} onclick={toggle} title="viewer settings" aria-label="viewer settings">
	<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
		<circle cx="12" cy="12" r="3" />
		<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
	</svg>
</button>

{#if open}
	<div class="vs-backdrop" onclick={() => (open = false)} role="presentation" use:portal></div>
	<div class="vs-menu" style="left:{anchor.x}px; top:{anchor.y}px" use:portal role="menu">
		{#if groups.length === 0}
			<div class="vs-empty">No settings</div>
		{/if}
		{#each groups as g (g.title)}
			<div class="grp">
				<button class="grp-head" onclick={() => toggleGroup(g.title)}>
					<span class="tri" class:shut={collapsed[g.title]}>▾</span>{g.title}
				</button>
				{#if !collapsed[g.title]}
					<div class="grp-body">
						{#each g.settings.filter(visible) as s (s.key)}
							<label class="row">
								<span class="lbl">{s.label}</span>
								{#if s.type === 'toggle'}
									<input
										type="checkbox"
										checked={settings[s.key] as boolean}
										onchange={(e) => set(s.key, e.currentTarget.checked)}
									/>
								{:else if s.type === 'select'}
									<select value={settings[s.key] as string} onchange={(e) => set(s.key, e.currentTarget.value)}>
										{#each s.options ?? [] as opt (opt)}<option value={opt}>{opt}</option>{/each}
									</select>
								{:else}
									<input
										type="number"
										value={settings[s.key] as number}
										min={s.min}
										max={s.max}
										step={s.step ?? 1}
										onchange={(e) => set(s.key, e.currentTarget.valueAsNumber)}
									/>
								{/if}
							</label>
						{/each}
					</div>
				{/if}
			</div>
		{/each}
	</div>
{/if}

<style>
	.cog {
		display: inline-grid;
		place-items: center;
		width: 16px;
		height: 16px;
		padding: 0;
		background: none;
		border: 0;
		color: var(--text-muted);
		cursor: pointer;
		transition: color 80ms ease;
	}
	.cog:hover,
	.cog.on {
		color: var(--text);
	}

	.vs-backdrop {
		position: fixed;
		inset: 0;
		z-index: calc(var(--z-menu) - 1);
	}
	.vs-menu {
		position: fixed;
		z-index: var(--z-menu);
		width: 212px;
		max-height: 70vh;
		overflow-y: auto;
		background: var(--surface-glass);
		backdrop-filter: blur(10px);
		border: 1px solid var(--border-strong);
		border-radius: var(--radius-sm);
		box-shadow: var(--shadow-2);
		font-family: var(--font-mono);
		padding: 4px;
	}
	.vs-empty {
		padding: 8px;
		font-size: 11px;
		color: var(--text-muted);
		text-align: center;
	}
	.grp + .grp {
		border-top: 1px solid var(--border);
		margin-top: 2px;
		padding-top: 2px;
	}
	.grp-head {
		display: flex;
		align-items: center;
		gap: 5px;
		width: 100%;
		padding: 4px 6px;
		background: transparent;
		border: 0;
		color: var(--text-dim);
		font-family: inherit;
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		cursor: pointer;
		border-radius: 3px;
	}
	.grp-head:hover {
		color: var(--text);
		background: var(--surface-2);
	}
	.tri {
		font-size: 9px;
		transition: transform 100ms ease;
	}
	.tri.shut {
		transform: rotate(-90deg);
	}
	.grp-body {
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding: 2px 2px 4px;
	}
	.row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		padding: 3px 6px;
		font-size: 11px;
		color: var(--text);
		border-radius: 3px;
	}
	.row:hover {
		background: var(--surface-2);
	}
	.lbl {
		color: var(--text-dim);
		white-space: nowrap;
	}
	.row input[type='number'],
	.row select {
		width: 84px;
		font-family: var(--font-mono);
		font-size: 11px;
		background: var(--bg);
		color: var(--text);
		border: 1px solid var(--border-strong);
		border-radius: 3px;
		padding: 2px 4px;
	}
	.row input[type='checkbox'] {
		accent-color: var(--accent);
		cursor: pointer;
	}
	.row input:focus,
	.row select:focus {
		outline: none;
		border-color: var(--accent);
	}
</style>
