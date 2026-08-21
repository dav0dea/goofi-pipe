<!-- Per-slot viewer settings: a cog opening a Popover of setting groups. Its role is `group`,
     not a menu — nothing in it is a menuitem. -->
<script lang="ts">
	import { settingsSchemaFor, type SettingDescriptor, type SettingValue } from './settingsSchema';
	import type { ViewBinding } from './viewBinding';
	import {
		Popover,
		Icon,
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
		// stopPropagation so picking the cog on a node header does not also toggle the slot's collapse.
		e.stopPropagation();
		open = !open;
	}

	function visible(s: SettingDescriptor): boolean {
		return !s.showWhen || settings[s.showWhen.key] === s.showWhen.equals;
	}
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
		<Icon name="settings" />
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
	/* The frozen node header wants the bare 16px glyph, not the 28px --hit box. Scoped through
	   `.vs-anchor` because a fully-`:global()` rule only TIES the primitive's, across CSS chunks. */
	.vs-anchor :global(.vs-cog) {
		min-width: 16px;
		min-height: 16px;
		color: var(--text-muted);
	}
	.vs-anchor :global(.vs-cog.on) {
		color: var(--text);
	}
	/* Hover half gated on real hover; the OPEN half above stays unconditional, as it reports state. */
	@media (hover: hover) {
		.vs-anchor :global(.vs-cog:hover) {
			color: var(--text);
		}
	}
	/* Real chrome lets the cog's box take the coarse floor; `--vs-cog-box` is the frozen 24px slot
	   header's opt-out, where a 44px box would hang out of the node. */
	@media (hover: none) and (pointer: coarse) {
		.vs-anchor :global(.vs-cog) {
			min-width: var(--vs-cog-box, var(--hit));
			min-height: var(--vs-cog-box, var(--hit));
		}
		.vs-anchor :global(.vs-cog::after) {
			inset: calc((var(--node-u) - 100%) / -2);
		}
	}
	:global(.vs-menu) {
		--popover-bg: var(--surface-glass);
		--popover-pad: var(--space-2);
		--popover-min-width: 0;
		/* This menu overhangs a small cog, which Popover's default --radius-md rounds visibly too much. */
		--popover-radius: var(--radius-sm);
		width: 212px;
		max-height: 70dvh;
		display: flex;
		flex-direction: column;
		backdrop-filter: blur(10px);
	}
</style>
