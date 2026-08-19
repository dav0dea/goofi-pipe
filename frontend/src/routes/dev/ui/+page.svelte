<!--
  /dev/ui — the primitive gallery harness (spec §6). A static, backend-free showcase of
  every `$lib/ui` primitive in its variants; the committed `tests/e2e` gallery specs drive
  it as the "failing test first" for a UI lib that has no vitest mount runner. Later P tasks
  extend this route + its specs. Each sample carries a distinct `data-testid`.
-->
<script lang="ts">
	import {
		Button,
		Icon,
		ICONS,
		IconButton,
		ScrollArea,
		Bar,
		Field,
		NumberInput,
		Slider,
		Select,
		TextInput,
		Trigger,
		Toggle,
		Tabs,
		Disclosure,
		Popover,
		Dialog,
		Badge,
		Chip,
		StatusDot,
		EmptyState,
		ChoiceGrid,
		type IconName,
		type ButtonVariant,
		type ButtonSize,
		type TabItem,
		type BadgeTone,
		type StatusTone,
		type StatusDotSize,
		type Choice
	} from '$lib/ui';

	let chosen = $state('—');
	const choices: Choice[] = [
		{ id: 'wave', label: 'Waveform', icon: 'activity', choose: () => (chosen = 'wave') },
		{ id: 'terminal', label: 'Terminal', icon: 'terminal', choose: () => (chosen = 'terminal') },
		{ id: 'plain', label: 'No icon of its own', choose: () => (chosen = 'plain') }
	];

	const variants: ButtonVariant[] = ['default', 'primary', 'ghost', 'danger'];
	const sizes: ButtonSize[] = ['sm', 'md'];
	// Display primitives (Task 6): every tone/size axis, driven from typed unions.
	const badgeTones: BadgeTone[] = ['neutral', 'accent', 'success', 'warning', 'danger'];
	const statusTones: StatusTone[] = ['ok', 'error', 'warn'];
	const dotSizes: StatusDotSize[] = ['sm', 'md'];
	// One icon per variant, from the app's single icon set — the gallery shows the primitive with
	// the content it actually carries in the product, not a stand-in glyph.
	const glyphs: Record<ButtonVariant, IconName> = {
		default: 'settings',
		primary: 'plus',
		ghost: 'refresh-cw',
		danger: 'x'
	};

	// --- Field-family demo state. Each control drives a piece of $state; the composition proves a
	// Slider + NumberInput share ONE value in a handful of lines. `*-committed` read-outs let the
	// e2e observe commit timing (a NumberInput must not update its committed value per keystroke). ---
	let gain = $state(1);
	let cutoff = $state(0.3);
	// The @container demo (Task 7): one shared value driven from a Slider + NumberInput placed as
	// DIRECT Field children, so `.ui-field-control`'s own flex-direction lays them out — and the
	// narrow-container query can flip it to a single column.
	let cqValue = $state(0.4);
	// A Slider seeded OUTSIDE its [min,max] so the track auto-extends (Slider.svelte:44-48) instead of
	// clipping — the e2e reads the range's extended min/max attributes.
	let sliderExtend = $state(5);
	// The swapping-control-region demo: ONE Field, three control kinds in an {#if} chain (the
	// inspector's fx toggle + ⤢ expand shape). The label's `for=` claim has to survive the swap.
	let swapMode = $state<'number' | 'text' | 'raw'>('number');
	let swapNum = $state(2);
	let swapText = $state('sin(x)');
	let refreshValue = $state('sine');
	let refreshing = $state(false);
	let refreshCount = $state(0);
	// A Select value that is NOT among its options — the stale-but-live case N's device/stream pickers
	// hit; Select.svelte:39 prepends it so it still renders selected.
	let stalePick = $state('unplugged-device');
	// An EMPTY current value: the truthy-guarded prepend must NOT add a blank leading option (N's
	// device pickers before a scan) — the empty-value rule `selectOptions` used to own, now at the P source.
	let emptyPick = $state('');
	// A `labels` map: the option value stays the raw key ('out0') while the dropdown shows a friendly label.
	let labelledPick = $state('out0');
	let textText = $state('hello');
	let textDecimal = $state('3.14');
	let textSearch = $state('');
	let textPath = $state('/home/user/patch.gfi');
	let triggerCount = $state(0);
	let toggled = $state(false);

	// Tabs: the connected bar drives which body renders below it (active tab merges into the body).
	const tabItems: TabItem[] = [
		{ id: 'signal', label: 'Signal' },
		{ id: 'audio', label: 'Audio' },
		{ id: 'video', label: 'Video' }
	];
	let activeTab = $state('signal');
	// Disclosure: bound open state, plus an onToggle read-out so the e2e can observe the callback.
	let disclosureOpen = $state(false);
	let disclosureToggles = $state(0);

	// --- Surfaces (Task 5). The Popover's trigger is anchored near the RIGHT edge so a naive
	// placement would overflow — proving `clampToViewport` shifts it back on-screen. ---
	let popoverAnchor = $state<HTMLElement | null>(null);
	let popoverOpen = $state(false);
	// A second sample proving the primitive imposes no role: this consumer declares role="menu"
	// via rest, and (nothing overriding it) that role reaches the surface.
	let menuPopoverAnchor = $state<HTMLElement | null>(null);
	let menuPopoverOpen = $state(false);
	let dialogOpen = $state(false);

	// Display: the Chip is the one pressable display primitive — a click read-out proves it fires.
	let chipCount = $state(0);

	function doRefresh(): void {
		// Simulate a device/stream re-scan: spin briefly, then land a fresh option set.
		refreshing = true;
		refreshCount += 1;
		setTimeout(() => {
			refreshing = false;
		}, 150);
	}
</script>

<main class="gallery">
	<h1>UI primitives</h1>

	<section>
		<h2>Button</h2>
		<div class="grid">
			{#each variants as variant (variant)}
				{#each sizes as size (size)}
					<Button {variant} {size} data-testid={`ui-button-${variant}-${size}`}>
						{variant} / {size}
					</Button>
				{/each}
			{/each}
			<Button variant="primary" size="md" disabled data-testid="ui-button-disabled">
				disabled
			</Button>
		</div>
	</section>

	<section>
		<h2>IconButton</h2>
		<div class="grid">
			{#each variants as variant (variant)}
				{#each sizes as size (size)}
					<IconButton
						{variant}
						{size}
						label={`${variant} ${size} action`}
						data-testid={`ui-icon-${variant}-${size}`}
					>
						<Icon name={glyphs[variant]} />
					</IconButton>
				{/each}
			{/each}
			<IconButton variant="default" size="md" disabled label="disabled action" data-testid="ui-icon-disabled">
				<Icon name={glyphs.default} />
			</IconButton>
		</div>
	</section>

	<section>
		<h2>Icon (the whole vendored set)</h2>
		<!-- Enumerated from `ICONS` rather than listed by hand, so the sheet cannot fall behind the
		     table: vendoring an icon puts it on this page, and an icon the app stopped drawing is
		     visible here as an orphan (`ui/icons.test.ts` fails it separately). Every tile inherits
		     its size from the surrounding type, which is the whole of Icon's sizing contract. -->
		<div class="grid" data-testid="ui-icon-set">
			{#each Object.keys(ICONS) as IconName[] as name (name)}
				<span class="icon-tile" title={name}><Icon {name} /></span>
			{/each}
		</div>
	</section>

	<section>
		<h2>ScrollArea</h2>
		<!-- A bounded flex frame; the ScrollArea fills it (flex:1 + min-height:0, which it owns) and
		     scrolls its overflowing content. The frame lives in the gallery's own markup so its height
		     is scoped here — the primitive itself stays unopinionated about its outer size. -->
		<div class="scroll-frame">
			<ScrollArea data-testid="ui-scrollarea">
				<div class="rows">
					{#each Array.from({ length: 30 }, (_, i) => i) as i (i)}
						<div class="box">row {i}</div>
					{/each}
				</div>
			</ScrollArea>
		</div>
	</section>

	<section>
		<h2>Bar</h2>
		<Bar data-testid="ui-bar">
			{#snippet start()}
				<span data-testid="ui-bar-start">Title</span>
			{/snippet}
			{#snippet end()}
				<div class="grid" data-testid="ui-bar-end">
					<Button size="sm">Save</Button>
					<IconButton size="sm" label="Settings"><Icon name="settings" /></IconButton>
				</div>
			{/snippet}
		</Bar>
	</section>

	<section>
		<h2>Field composition (the north star)</h2>
		<!-- The proof: a Slider + NumberInput sharing ONE `cutoff` value inside a labelled Field, in
		     a handful of lines. Moving either drives the other; the label focuses the first control. -->
		<div class="form">
			<Field label="cutoff" data-testid="ui-compose-field">
				<Slider
					value={cutoff}
					onChange={(v) => (cutoff = v)}
					min={0}
					max={1}
					step={0.01}
					data-testid="ui-compose-slider"
				/>
				<NumberInput
					value={cutoff}
					onChange={(v) => (cutoff = v)}
					min={0}
					max={1}
					step={0.01}
					scrub
					data-testid="ui-compose-number"
				/>
			</Field>
			<span class="readout" data-testid="ui-compose-value">{cutoff}</span>
		</div>
	</section>

	<section>
		<h2>Field + NumberInput (commit-on-blur)</h2>
		<!-- A single-control Field: clicking the label focuses the input (real <label for>), and the
		     committed value updates on blur/Enter, NOT per keystroke — both asserted by the e2e. -->
		<div class="form">
			<Field label="gain" data-testid="ui-field-single">
				<NumberInput value={gain} onChange={(v) => (gain = v)} data-testid="ui-field-number" />
			</Field>
			<span class="readout" data-testid="ui-field-committed">{gain}</span>
		</div>
	</section>

	{#snippet swapChips()}
		<Chip
			tone={swapMode === 'number' ? 'neutral' : 'accent'}
			onclick={() => (swapMode = swapMode === 'number' ? 'text' : 'number')}
			data-testid="ui-field-swap-fx">fx</Chip
		>
		<Chip
			onclick={() => (swapMode = swapMode === 'raw' ? 'text' : 'raw')}
			data-testid="ui-field-swap-expand"><Icon name="maximize-2" /></Chip
		>
	{/snippet}

	<section>
		<h2>Field with a swapping control region (the ParamField shape)</h2>
		<!-- ONE persistent Field whose control region is an {#if} chain flipped by an adornment Chip —
		     exactly how the inspector's fx toggle and its ⤢ expand behave. The label's `for=` must
		     follow whichever control is actually mounted, in BOTH directions, and must come back after
		     a detour through a raw control that claims nothing (the multi-line textarea's shape). -->
		<div class="form">
			<Field label="swap" data-testid="ui-field-swap" adornment={swapChips}>
				{#if swapMode === 'number'}
					<NumberInput
						value={swapNum}
						onChange={(v) => (swapNum = v)}
						data-testid="ui-field-swap-number"
					/>
				{:else if swapMode === 'text'}
					<TextInput
						value={swapText}
						onChange={(v) => (swapText = v)}
						data-testid="ui-field-swap-text"
					/>
				{:else}
					<!-- Claims nothing, like ParamField's expanded multi-line editor; it carries its own name. -->
					<textarea
						rows="2"
						aria-label="swap raw"
						bind:value={swapText}
						data-testid="ui-field-swap-raw"
					></textarea>
				{/if}
			</Field>
		</div>
	</section>

	<section>
		<h2>Slider</h2>
		<div class="form">
			<Slider
				value={cutoff}
				onChange={(v) => (cutoff = v)}
				min={0}
				max={1}
				step={0.01}
				data-testid="ui-slider"
			/>
			<!-- Seeded at 5 on a [0,1] track: the thumb renders in range because the track auto-extends its
			     min/max to span the live value instead of clipping at the edge. -->
			<Slider
				value={sliderExtend}
				onChange={(v) => (sliderExtend = v)}
				min={0}
				max={1}
				data-testid="ui-slider-extend"
			/>
		</div>
	</section>

	<section>
		<h2>Select (with refresh)</h2>
		<div class="form">
			<Field label="waveform" data-testid="ui-select-field">
				<Select
					value={refreshValue}
					onChange={(v) => (refreshValue = v)}
					options={['sine', 'square', 'saw', 'triangle']}
					onRefresh={doRefresh}
					{refreshing}
					data-testid="ui-select"
				/>
			</Field>
			<span class="readout" data-testid="ui-select-refreshes">{refreshCount}</span>
			<!-- The committed value the <select>'s onchange emits — the e2e reads it to prove onChange fires. -->
			<span class="readout" data-testid="ui-select-value">{refreshValue}</span>
			<!-- A value NOT in its options (a stale-but-live device id): it still renders selected because
			     the Select prepends the current value to the list. -->
			<Field label="stale value" data-testid="ui-select-stale-field">
				<Select
					value={stalePick}
					onChange={(v) => (stalePick = v)}
					options={['sine', 'square', 'saw', 'triangle']}
					data-testid="ui-select-stale"
				/>
			</Field>
			<!-- An EMPTY current value: the truthy-guarded prepend leaves the list untouched, so there is
			     no blank leading option (unlike a naive `includes` check, which would prepend ''). -->
			<Field label="empty value" data-testid="ui-select-empty-field">
				<Select
					value={emptyPick}
					onChange={(v) => (emptyPick = v)}
					options={['sine', 'square', 'saw', 'triangle']}
					data-testid="ui-select-empty"
				/>
			</Field>
			<!-- A `labels` map: the committed value stays the raw option key while the dropdown shows a
			     friendlier label (e.g. a sub-patch port's user name + dtype). -->
			<Field label="labelled" data-testid="ui-select-labelled-field">
				<Select
					value={labelledPick}
					onChange={(v) => (labelledPick = v)}
					options={['out0', 'out1']}
					labels={{ out0: 'envelope · array', out1: 'trigger · array' }}
					data-testid="ui-select-labelled"
				/>
			</Field>
			<span class="readout" data-testid="ui-select-labelled-value">{labelledPick}</span>
		</div>
	</section>

	<section>
		<h2>TextInput (inputmode variants)</h2>
		<div class="form">
			<Field label="text">
				<TextInput value={textText} onChange={(v) => (textText = v)} inputmode="text" data-testid="ui-text-text" />
			</Field>
			<Field label="decimal">
				<TextInput
					value={textDecimal}
					onChange={(v) => (textDecimal = v)}
					inputmode="decimal"
					data-testid="ui-text-decimal"
				/>
			</Field>
			<Field label="search">
				<TextInput
					value={textSearch}
					onChange={(v) => (textSearch = v)}
					inputmode="search"
					placeholder="search…"
					data-testid="ui-text-search"
				/>
			</Field>
			<Field label="path">
				<TextInput value={textPath} onChange={(v) => (textPath = v)} inputmode="path" data-testid="ui-text-path" />
			</Field>
			<!-- The committed value of the text field — onChange fires on blur/Enter, NOT per keystroke, so
			     the e2e reads it to observe commit timing (typing buffers; blur/Enter commits). -->
			<span class="readout" data-testid="ui-text-committed">{textText}</span>
		</div>
	</section>

	<section>
		<h2>Trigger</h2>
		<div class="form">
			<Trigger onclick={() => (triggerCount += 1)} data-testid="ui-trigger">reset</Trigger>
			<span class="readout" data-testid="ui-trigger-count">{triggerCount}</span>
		</div>
	</section>

	<section>
		<h2>Toggle</h2>
		<div class="form">
			<Field label="enabled" data-testid="ui-toggle-field">
				<Toggle value={toggled} onChange={(v) => (toggled = v)} data-testid="ui-toggle" />
			</Field>
			<span class="readout" data-testid="ui-toggle-value">{toggled ? 'on' : 'off'}</span>
		</div>
	</section>

	<section>
		<h2>Tabs (the connected bar)</h2>
		<!-- The active tab drops to the body surface and merges with the panel body flush beneath it —
		     the `.tabs-body` paints the SAME `--surface-1` the active tab drops to (no line between
		     them). The e2e asserts the active tab's computed background equals the body's (connected),
		     and differs from an inactive tab's (the drop happened). -->
		<div class="tabs-demo">
			<Tabs items={tabItems} active={activeTab} onSelect={(id) => (activeTab = id)} data-testid="ui-tabs" />
			<div class="tabs-body" data-testid="ui-tabs-body">
				<span data-testid="ui-tabs-active">{activeTab}</span> panel content
			</div>
		</div>
	</section>

	<section>
		<h2>Disclosure</h2>
		<div class="form">
			<Disclosure
				bind:open={disclosureOpen}
				onToggle={() => (disclosureToggles += 1)}
				data-testid="ui-disclosure"
			>
				{#snippet summary()}
					Advanced options
				{/snippet}
				<p class="disclosure-content" data-testid="ui-disclosure-content">
					Collapsed by default; the caret rotates and this region mounts on toggle.
				</p>
			</Disclosure>
			<span class="readout" data-testid="ui-disclosure-toggles">{disclosureToggles}</span>
		</div>
	</section>

	<section>
		<h2>Popover (anchored + clamped)</h2>
		<!-- The trigger hugs the right edge so a naive drop would overflow; the Popover's clamp shifts
		     it back on-screen (the e2e asserts its box stays within innerWidth/innerHeight). Escape and
		     an outside pointerdown both dismiss it. -->
		<div class="pop-row">
			<span class="pop-anchor" bind:this={popoverAnchor}>
				<Button onclick={() => (popoverOpen = !popoverOpen)} data-testid="ui-popover-trigger">
					{popoverOpen ? 'Close' : 'Open'} popover
				</Button>
			</span>
		</div>
		<Popover
			anchor={popoverAnchor}
			open={popoverOpen}
			onDismiss={() => (popoverOpen = false)}
			data-testid="ui-popover"
		>
			<div class="pop-content" data-testid="ui-popover-content">
				<strong>Anchored overlay</strong>
				<p>Portalled, clamped on-screen, self-dismissing on Escape or an outside click.</p>
			</div>
		</Popover>

		<!-- Same primitive, but the consumer declares its own semantics: `role="menu"` flows through
		     `...rest` and reaches the surface, proving the Popover forces no role of its own. -->
		<div class="pop-row">
			<span class="pop-anchor" bind:this={menuPopoverAnchor}>
				<Button
					onclick={() => (menuPopoverOpen = !menuPopoverOpen)}
					data-testid="ui-menu-popover-trigger"
				>
					{menuPopoverOpen ? 'Close' : 'Open'} menu popover
				</Button>
			</span>
		</div>
		<Popover
			anchor={menuPopoverAnchor}
			open={menuPopoverOpen}
			onDismiss={() => (menuPopoverOpen = false)}
			role="menu"
			data-testid="ui-menu-popover"
		>
			<div class="pop-content" data-testid="ui-menu-popover-content">
				<strong>Menu-role overlay</strong>
				<p>The consumer passes role="menu" via rest; the primitive imposes nothing over it.</p>
			</div>
		</Popover>
	</section>

	<section>
		<h2>Dialog (centered + focus-trap)</h2>
		<div class="form">
			<Button onclick={() => (dialogOpen = true)} data-testid="ui-dialog-trigger">Open dialog</Button>
		</div>
		<Dialog open={dialogOpen} onClose={() => (dialogOpen = false)} data-testid="ui-dialog">
			<div class="dialog-content" data-testid="ui-dialog-content">
				<h3>Confirm action</h3>
				<p>A centered modal: focus is trapped inside, Escape and a backdrop click both close it.</p>
				<!-- Deliberately taller than `--dialog-max-height`, so the dialog paints its OWN
				     scrollbar: a click there targets the dialog element exactly like a backdrop click
				     does, and the e2e pins that it does not dismiss. -->
				<p class="filler">
					{#each { length: 40 } as _, i (i)}
						Overflowing line {i + 1} — the dialog scrolls its own content.
					{/each}
				</p>
				<div class="grid end">
					<Button onclick={() => (dialogOpen = false)} data-testid="ui-dialog-cancel">Cancel</Button>
					<Button variant="primary" onclick={() => (dialogOpen = false)} data-testid="ui-dialog-confirm">
						Confirm
					</Button>
				</div>
			</div>
		</Dialog>
	</section>

	<section>
		<h2>Badge (static tone pill)</h2>
		<!-- The five tones; success shares the accent token by design, so the e2e checks the four
		     meaningfully-distinct tones resolve to distinct text colours. -->
		<div class="grid">
			{#each badgeTones as tone (tone)}
				<Badge {tone} data-testid={`ui-badge-${tone}`}>{tone}</Badge>
			{/each}
		</div>
	</section>

	<section>
		<h2>Chip (pressable tone pill)</h2>
		<div class="grid">
			{#each badgeTones as tone (tone)}
				<Chip {tone} data-testid={`ui-chip-${tone}`}>{tone}</Chip>
			{/each}
		</div>
		<div class="form">
			<Chip tone="accent" onclick={() => (chipCount += 1)} data-testid="ui-chip">click me</Chip>
			<span class="readout" data-testid="ui-chip-count">{chipCount}</span>
		</div>
	</section>

	<section>
		<h2>StatusDot (no glow)</h2>
		<!-- Each tone is a plain filled circle — NO box-shadow halo (the health-dot regression guard). -->
		<div class="grid">
			{#each statusTones as tone (tone)}
				<div class="dot-label">
					<StatusDot {tone} data-testid={`ui-statusdot-${tone}`} />
					<span class="readout">{tone}</span>
				</div>
			{/each}
		</div>
		<div class="grid">
			{#each dotSizes as size (size)}
				<StatusDot tone="ok" {size} data-testid={`ui-statusdot-size-${size}`} />
			{/each}
		</div>
		<!-- `pulse` blinks the dot, for a state that is not merely bad but GONE (a dead node). -->
		<div class="grid">
			<StatusDot tone="error" pulse data-testid="ui-statusdot-pulse" />
		</div>
	</section>

	<section>
		<h2>ChoiceGrid (icon-over-label tiles)</h2>
		<!-- The grid an empty surface offers its choices with — the empty panel's and the agent
		     panel's launcher. A tile with no icon of its own falls back to `square-dashed`, and the
		     readout proves `choose` fires (the tiles are the affordance, not a wrapper's click). -->
		<ChoiceGrid {choices} data-testid="ui-choicegrid" />
		<div class="readout" data-testid="ui-choicegrid-value">{chosen}</div>
	</section>

	<section>
		<h2>EmptyState (centred placeholder)</h2>
		<!-- Full: title + hint, centred on both axes inside a bounded frame. -->
		<div class="empty-frame">
			<EmptyState data-testid="ui-emptystate">
				{#snippet title()}No nodes yet{/snippet}
				{#snippet hint()}Add a node from the palette to get started.{/snippet}
			</EmptyState>
		</div>
		<!-- Bare: no snippets — must still render as a valid (empty) centred box. -->
		<div class="empty-frame">
			<EmptyState data-testid="ui-emptystate-bare" />
		</div>
	</section>

	<section>
		<h2>Field in a narrow @container (single-column stack)</h2>
		<!-- The `@container` enablement (Task 7): the Field's control row stacks to a single column
		     when its query container is narrower than the threshold, and stays a row when wide. Each
		     wrapper is its OWN `container-type: inline-size` context standing in for a narrow vs wide
		     panel body (the real query container is `.panel-body`), so the SAME Field responds to its
		     container's width. The Slider + NumberInput are DIRECT Field children so `.ui-field-control`
		     itself lays them out. -->
		<div class="cq-box cq-narrow" data-testid="ui-cq-narrow">
			<Field label="cutoff" data-testid="ui-cq-narrow-field">
				<Slider
					value={cqValue}
					onChange={(v) => (cqValue = v)}
					min={0}
					max={1}
					step={0.01}
					data-testid="ui-cq-narrow-slider"
				/>
				<NumberInput
					value={cqValue}
					onChange={(v) => (cqValue = v)}
					min={0}
					max={1}
					step={0.01}
					data-testid="ui-cq-narrow-number"
				/>
			</Field>
		</div>
		<div class="cq-box cq-wide" data-testid="ui-cq-wide">
			<Field label="cutoff" data-testid="ui-cq-wide-field">
				<Slider
					value={cqValue}
					onChange={(v) => (cqValue = v)}
					min={0}
					max={1}
					step={0.01}
					data-testid="ui-cq-wide-slider"
				/>
				<NumberInput
					value={cqValue}
					onChange={(v) => (cqValue = v)}
					min={0}
					max={1}
					step={0.01}
					data-testid="ui-cq-wide-number"
				/>
			</Field>
		</div>
	</section>
</main>

<style>
	.gallery {
		box-sizing: border-box;
		height: 100vh;
		overflow-y: auto;
		padding: var(--space-8);
		display: flex;
		flex-direction: column;
		gap: var(--space-8);
	}
	h1 {
		margin: 0;
		font-size: var(--fs-title);
		color: var(--text);
	}
	h2 {
		margin: 0 0 var(--space-4);
		font-size: var(--fs-strong);
		color: var(--text-dim);
	}
	.grid {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: var(--space-6);
	}
	.grid.end {
		justify-content: flex-end;
	}
	/* One icon of the sheet. Icon sizes at 1em, so the tile's own type size IS the icon size —
	   the sheet reads at the display scale while the product's copies stay at their control's. */
	.icon-tile {
		display: flex;
		font-size: var(--fs-title);
		color: var(--text-dim);
	}
	/* A status dot beside its name. */
	.dot-label {
		display: flex;
		align-items: center;
		gap: var(--space-2);
	}
	/* The ScrollArea sample's overflowing content. */
	.rows {
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
	}
	/* Gallery-local demo children — plain boxes so a sample's own spacing is what's measured. */
	.box {
		display: grid;
		place-items: center;
		padding: var(--space-2) var(--space-4);
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
	}
	/* Bounded frame the ScrollArea fills, so its overflowing content scrolls. */
	.scroll-frame {
		display: flex;
		flex-direction: column;
		height: 8rem;
		width: 12rem;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		padding: var(--space-4);
	}
	/* Bounded column the Field-family samples live in — a panel-width form so the labelled controls
	   size realistically (they fill their Field). */
	.form {
		display: flex;
		flex-direction: column;
		gap: var(--space-6);
		width: 18rem;
		max-width: 100%;
	}
	/* A committed-value read-out the e2e reads to observe commit timing. */
	.readout {
		font-size: var(--fs-micro);
		color: var(--text-muted);
		font-variant-numeric: tabular-nums;
	}
	/* The Tabs demo panel: the body paints the SAME surface the active tab drops to (--surface-1,
	   the Tabs `--tabs-body` default), so the active tab merges into it seamlessly. */
	.tabs-demo {
		width: 22rem;
		max-width: 100%;
	}
	.tabs-body {
		padding: var(--space-7);
		background: var(--surface-1);
		border-radius: 0 0 var(--radius-sm) var(--radius-sm);
		color: var(--text-dim);
		font-size: var(--fs-small);
	}
	.disclosure-content {
		margin: 0;
		color: var(--text-dim);
		font-size: var(--fs-small);
	}
	/* Push the Popover trigger to the right edge so its overflow clamp is exercised. */
	.pop-row {
		display: flex;
		justify-content: flex-end;
	}
	.pop-anchor {
		display: inline-flex;
	}
	.pop-content {
		display: flex;
		flex-direction: column;
		gap: var(--space-3);
		max-width: 16rem;
	}
	.pop-content p,
	.dialog-content p {
		margin: 0;
		color: var(--text-dim);
		font-size: var(--fs-small);
	}
	.dialog-content {
		display: flex;
		flex-direction: column;
		gap: var(--space-6);
	}
	.dialog-content .filler {
		color: var(--text-muted);
	}
	.dialog-content h3 {
		margin: 0;
		font-size: var(--fs-strong);
		color: var(--text);
	}
	/* Query-container demo wrappers. Each establishes its OWN inline-size container so the Field's
	   `@container` rule resolves against it. The widths are structural px chosen to sit either side of
	   the 240px threshold (like the threshold itself, an allowed structural literal). */
	.cq-box {
		container-type: inline-size;
		box-sizing: border-box;
		margin-top: var(--space-4);
		padding: var(--space-4);
		border: 1px dashed var(--border);
		border-radius: var(--radius-sm);
	}
	.cq-narrow {
		width: 200px;
	}
	.cq-wide {
		width: 360px;
	}
	/* Bounded frame the EmptyState fills (single grid cell → stretches both axes), so its own
	   both-axis centering is what's measured. */
	.empty-frame {
		display: grid;
		height: 10rem;
		width: 20rem;
		max-width: 100%;
		margin-top: var(--space-4);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
	}
</style>
