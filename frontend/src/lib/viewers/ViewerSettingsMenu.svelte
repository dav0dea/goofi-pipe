<!--
  Per-slot viewer settings — a cog button that opens a Blender-style dropdown of
  collapsible setting groups, driven by the viewer's settings schema. The dropdown
  is the shared `Popover` (portalled to <body> so it escapes the zoomed/clipped
  SvelteFlow viewport, clamped on-screen, self-dismissing on Escape or an outside
  click); each group is a `Disclosure`; each setting is a `Field` + the control its
  type maps to. Writes still go straight to the binding, which owns persistence.

  It reads as a menu and is not one: nothing in it is a menuitem, so it declares the named `group`
  it actually is — `Popover` imposes no role precisely so the consumer can be accurate — and the
  cog carries the open state a sighted user reads off the surface simply being there.
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
		aria-expanded={open}
		data-testid="viewer-settings-cog"
		onclick={toggle}
	>
		<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
			<circle cx="12" cy="12" r="3" />
			<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
		</svg>
	</IconButton>
</span>

<Popover
	{anchor}
	{open}
	onDismiss={() => (open = false)}
	catcher
	role="group"
	aria-label="viewer settings"
	class="vs-menu"
	data-testid="viewer-settings-menu"
>
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
	   bare 16px glyph (not the 28px --hit box), so the visual box is pinned back to 16px.
	   Muted at rest, brightening on hover / while open. What a coarse pointer gets is
	   decided by the block below, per host.

	   Scoped through `.vs-anchor` instead of reaching for `.ui-icon-btn`: a fully-`:global()`
	   `.ui-icon-btn.vs-cog` scores exactly what the primitive's own `.ui-icon-btn.svelte-xxx`
	   scores, and the two land in different built CSS chunks where a tie is settled by the emitted
	   <link> order rather than by the source (the hazard `54de8a1` fixed for `.content-btn`). The
	   anchor's scope class carries these above the tie in source.

	   Deliberately NOT `density="chrome"` + `--icon-btn-size`: that density restores the box to
	   --hit under a coarse pointer, which would stand a 44px cog inside this frozen 24px --node-u
	   header (touch-floor.spec pins that). */
	.vs-anchor :global(.vs-cog) {
		min-width: 16px;
		min-height: 16px;
		color: var(--text-muted);
	}
	.vs-anchor :global(.vs-cog:hover),
	.vs-anchor :global(.vs-cog.on) {
		color: var(--text);
	}
	/* R closes the other half (C27). Two hosts, two answers, one hook.
	   In REAL chrome — a docked ViewerPanel header, --hit tall, where the Unlink button and the kind
	   select both take the floor — the cog's own BOX takes it too. A box lays out, so it cannot
	   reach over the select 6px to its left the way a positioned pseudo would (C18); IconButton's
	   coarse `::after` then adds nothing, which is exactly right.
	   `--vs-cog-box` is the frozen host's opt-out: SlotViewer's 24px `--node-u` slot header, where a
	   44px box is not a bigger target but a control hanging out of the node with its bottom half
	   clipped. There the carve-out below is what the strip can offer instead — bounded to the
	   header's own unit, so it cannot reach the rows above and below and take their taps, and it
	   cannot touch a neighbour either (the flex gaps are wider than the 4px it grows). Bounding it
	   to `--node-u` unconditionally is what left the docked cog at 24×24: `--node-u` is a `:root`
	   token, so the frozen host's answer applied in both. */
	@media (hover: none) and (pointer: coarse) {
		.vs-anchor :global(.vs-cog) {
			min-width: var(--vs-cog-box, var(--hit));
			min-height: var(--vs-cog-box, var(--hit));
		}
		.vs-anchor :global(.vs-cog::after) {
			inset: calc((var(--node-u) - 100%) / -2);
		}
	}
	/* The dropdown surface: the shared Popover, tuned to the compact glassy menu it was —
	   a fixed 212px column that scrolls its groups within 70dvh. */
	:global(.vs-menu) {
		--popover-bg: var(--surface-glass);
		--popover-pad: var(--space-2);
		--popover-min-width: 0;
		/* This menu overhangs a small cog inside the viewer chrome, not a wide panel — Popover's
		   default --radius-md rounds it visibly more than the pre-migration menu did (C13). */
		--popover-radius: var(--radius-sm);
		width: 212px;
		max-height: 70dvh;
		display: flex;
		flex-direction: column;
		backdrop-filter: blur(10px);
		font-family: var(--font-mono);
	}
</style>
