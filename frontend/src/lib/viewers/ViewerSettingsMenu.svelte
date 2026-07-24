<!--
  Per-slot viewer settings — a cog button that opens a Blender-style dropdown of
  collapsible setting groups, driven by the viewer's settings schema. The dropdown
  is the shared `Popover` (portalled to <body> so it escapes the zoomed/clipped
  SvelteFlow viewport, clamped on-screen, self-dismissing on Escape or an outside
  click); each group is a `Disclosure`; each setting is a `Field` + the control its
  type maps to. Writes still go straight to the binding, which owns persistence.
-->
<script lang="ts">
	import { settingsSchemaFor, type SettingDescriptor, type SettingValue } from './settingsSchema';
	import type { ViewBinding } from './viewBinding';
	import {
		Popover,
		IconButton,
		Disclosure,
		Field,
		ScrollArea,
		EmptyState,
		Toggle,
		Select,
		NumberInput
	} from '$lib/ui';

	let { binding }: { binding: ViewBinding } = $props();

	const kind = $derived(binding.kind);
	const groups = $derived(settingsSchemaFor(kind));
	const settings = $derived(binding.settings);

	let open = $state(false);
	let anchor = $state<HTMLElement | null>(null);
	let collapsed = $state<Record<string, boolean>>({});

	function toggle(e: MouseEvent): void {
		// stopPropagation so picking the cog on a node header doesn't also toggle the
		// slot's collapse; harmless in the panel header.
		e.stopPropagation();
		open = !open;
	}

	function visible(s: SettingDescriptor): boolean {
		return !s.showWhen || settings[s.showWhen.key] === s.showWhen.equals;
	}
	// Writes go straight to the binding, which owns persistence (inline →
	// node.viewers; panel → layout).
	function set(key: string, value: SettingValue): void {
		binding.setSetting(key, value);
	}
</script>

<span class="vs-anchor" bind:this={anchor}>
	<IconButton
		variant="ghost"
		size="sm"
		class={open ? 'vs-cog on' : 'vs-cog'}
		label="viewer settings"
		onclick={toggle}
	>
		<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
			<circle cx="12" cy="12" r="3" />
			<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
		</svg>
	</IconButton>
</span>

<Popover {anchor} {open} onDismiss={() => (open = false)} role="menu" class="vs-menu">
	<ScrollArea>
		{#if groups.length === 0}
			<EmptyState>
				{#snippet hint()}No settings{/snippet}
			</EmptyState>
		{/if}
		{#each groups as g (g.title)}
			<Disclosure
				open={!collapsed[g.title]}
				onToggle={(o) => (collapsed = { ...collapsed, [g.title]: !o })}
			>
				{#snippet summary()}{g.title}{/snippet}
				{#each g.settings.filter(visible) as s (s.key)}
					<Field label={s.label}>
						{#if s.type === 'toggle'}
							<Toggle value={settings[s.key] as boolean} onChange={(v) => set(s.key, v)} />
						{:else if s.type === 'select'}
							<Select
								options={s.options ?? []}
								value={settings[s.key] as string}
								onChange={(v) => set(s.key, v)}
							/>
						{:else}
							<NumberInput
								value={settings[s.key] as number}
								onChange={(v) => set(s.key, v)}
								min={s.min}
								max={s.max}
								step={s.step ?? 1}
							/>
						{/if}
					</Field>
				{/each}
			</Disclosure>
		{/each}
	</ScrollArea>
</Popover>

<style>
	.vs-anchor {
		display: inline-flex;
	}
	/* The cog is the app's icon-button, but the frozen node header wants the original
	   bare 16px glyph (not the 28px --hit box), so the visual box is pinned back to 16px
	   — the coarse-pointer ::after still guarantees a --hit tap target. Muted at rest,
	   brightening on hover / while open. */
	:global(.ui-icon-btn.vs-cog) {
		min-width: 16px;
		min-height: 16px;
		color: var(--text-muted);
	}
	:global(.ui-icon-btn.vs-cog:hover),
	:global(.ui-icon-btn.vs-cog.on) {
		color: var(--text);
	}
	/* The dropdown surface: the shared Popover, tuned to the compact glassy menu it was —
	   a fixed 212px column that scrolls its groups within 70vh. */
	:global(.vs-menu) {
		--popover-bg: var(--surface-glass);
		--popover-pad: var(--space-2);
		--popover-min-width: 0;
		width: 212px;
		max-height: 70vh;
		display: flex;
		flex-direction: column;
		backdrop-filter: blur(10px);
		font-family: var(--font-mono);
	}
</style>
