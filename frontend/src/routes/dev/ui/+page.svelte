<!-- /dev/ui — gallery of every `$lib/ui` primitive; the `tests/e2e` gallery specs drive it. -->
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
	const badgeTones: BadgeTone[] = ['neutral', 'accent', 'success', 'warning', 'danger'];
	const statusTones: StatusTone[] = ['ok', 'error', 'warn'];
	const dotSizes: StatusDotSize[] = ['sm', 'md'];
	const glyphs: Record<ButtonVariant, IconName> = {
		default: 'settings',
		primary: 'plus',
		ghost: 'refresh-cw',
		danger: 'x'
	};

	let gain = $state(1);
	let cutoff = $state(0.3);
	let cqValue = $state(0.4);
	let sliderExtend = $state(5);
	let swapMode = $state<'number' | 'text' | 'raw'>('number');
	let swapNum = $state(2);
	let swapText = $state('sin(x)');
	let refreshValue = $state('sine');
	let refreshing = $state(false);
	let refreshCount = $state(0);
	let stalePick = $state('unplugged-device');
	let emptyPick = $state('');
	let labelledPick = $state('out0');
	let textText = $state('hello');
	let textDecimal = $state('3.14');
	let textSearch = $state('');
	let textPath = $state('/home/user/patch.gfi');
	let triggerCount = $state(0);
	let toggled = $state(false);

	const tabItems: TabItem[] = [
		{ id: 'signal', label: 'Signal' },
		{ id: 'audio', label: 'Audio' },
		{ id: 'video', label: 'Video' }
	];
	let activeTab = $state('signal');
	let disclosureOpen = $state(false);
	let disclosureToggles = $state(0);

	let popoverAnchor = $state<HTMLElement | null>(null);
	let popoverOpen = $state(false);
	let menuPopoverAnchor = $state<HTMLElement | null>(null);
	let menuPopoverOpen = $state(false);
	let dialogOpen = $state(false);

	let chipCount = $state(0);

	function doRefresh(): void {
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
		<div class="grid" data-testid="ui-icon-set">
			{#each Object.keys(ICONS) as IconName[] as name (name)}
				<span class="icon-tile" title={name}><Icon {name} /></span>
			{/each}
		</div>
	</section>

	<section>
		<h2>ScrollArea</h2>
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
			<span class="readout" data-testid="ui-select-value">{refreshValue}</span>
			<Field label="stale value" data-testid="ui-select-stale-field">
				<Select
					value={stalePick}
					onChange={(v) => (stalePick = v)}
					options={['sine', 'square', 'saw', 'triangle']}
					data-testid="ui-select-stale"
				/>
			</Field>
			<Field label="empty value" data-testid="ui-select-empty-field">
				<Select
					value={emptyPick}
					onChange={(v) => (emptyPick = v)}
					options={['sine', 'square', 'saw', 'triangle']}
					data-testid="ui-select-empty"
				/>
			</Field>
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
		<div class="grid">
			<StatusDot tone="error" pulse data-testid="ui-statusdot-pulse" />
		</div>
	</section>

	<section>
		<h2>ChoiceGrid (icon-over-label tiles)</h2>
		<ChoiceGrid {choices} data-testid="ui-choicegrid" />
		<div class="readout" data-testid="ui-choicegrid-value">{chosen}</div>
	</section>

	<section>
		<h2>EmptyState (centred placeholder)</h2>
		<div class="empty-frame">
			<EmptyState data-testid="ui-emptystate">
				{#snippet title()}No nodes yet{/snippet}
				{#snippet hint()}Add a node from the palette to get started.{/snippet}
			</EmptyState>
		</div>
		<div class="empty-frame">
			<EmptyState data-testid="ui-emptystate-bare" />
		</div>
	</section>

	<section>
		<h2>Field in a narrow @container (single-column stack)</h2>
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
	.icon-tile {
		display: flex;
		font-size: var(--fs-title);
		color: var(--text-dim);
	}
	.dot-label {
		display: flex;
		align-items: center;
		gap: var(--space-2);
	}
	.rows {
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
	}
	.box {
		display: grid;
		place-items: center;
		padding: var(--space-2) var(--space-4);
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
	}
	.scroll-frame {
		display: flex;
		flex-direction: column;
		height: 8rem;
		width: 12rem;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		padding: var(--space-4);
	}
	.form {
		display: flex;
		flex-direction: column;
		gap: var(--space-6);
		width: 18rem;
		max-width: 100%;
	}
	.readout {
		font-size: var(--fs-micro);
		color: var(--text-muted);
		font-variant-numeric: tabular-nums;
	}
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
	/* Widths sit either side of the Field's 240px @container threshold. */
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
