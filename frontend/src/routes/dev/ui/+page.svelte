<!--
  /dev/ui — the primitive gallery harness (spec §6). A static, backend-free showcase of
  every `$lib/ui` primitive in its variants; the committed `tests/e2e` gallery specs drive
  it as the "failing test first" for a UI lib that has no vitest mount runner. Later P tasks
  extend this route + its specs. Each sample carries a distinct `data-testid`.
-->
<script lang="ts">
	import {
		Button,
		IconButton,
		Stack,
		Row,
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
		PanelShell,
		Badge,
		Chip,
		StatusDot,
		Spinner,
		EmptyState,
		type ButtonVariant,
		type ButtonSize,
		type TabItem,
		type BadgeTone,
		type StatusTone,
		type StatusDotSize,
		type SpinnerSize
	} from '$lib/ui';

	const variants: ButtonVariant[] = ['default', 'primary', 'ghost', 'danger'];
	const sizes: ButtonSize[] = ['sm', 'md'];
	// Display primitives (Task 6): every tone/size axis, driven from typed unions.
	const badgeTones: BadgeTone[] = ['neutral', 'accent', 'success', 'warning', 'danger'];
	const statusTones: StatusTone[] = ['ok', 'error', 'warn'];
	const dotSizes: StatusDotSize[] = ['sm', 'md'];
	const spinnerSizes: SpinnerSize[] = ['sm', 'md'];
	// Single-character glyphs so the IconButton glyph stays visibly small.
	const glyphs: Record<ButtonVariant, string> = {
		default: '⚙', // gear
		primary: '+',
		ghost: '⟳', // refresh
		danger: '×' // multiplication sign
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
	let refreshValue = $state('sine');
	let refreshing = $state(false);
	let refreshCount = $state(0);
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
						{glyphs[variant]}
					</IconButton>
				{/each}
			{/each}
			<IconButton variant="default" size="md" disabled label="disabled action" data-testid="ui-icon-disabled">
				{glyphs.default}
			</IconButton>
		</div>
	</section>

	<section>
		<h2>Row</h2>
		<!-- gap=4 (the default) → var(--space-4); the e2e measures the inter-child gap against the token. -->
		<Row gap={4} data-testid="ui-row-gap4">
			<div class="box" data-testid="ui-row-child-a">A</div>
			<div class="box" data-testid="ui-row-child-b">B</div>
			<div class="box">C</div>
		</Row>
	</section>

	<section>
		<h2>Stack</h2>
		<Stack gap={4} data-testid="ui-stack-gap4">
			<div class="box" data-testid="ui-stack-child-a">A</div>
			<div class="box" data-testid="ui-stack-child-b">B</div>
			<div class="box">C</div>
		</Stack>
	</section>

	<section>
		<h2>ScrollArea</h2>
		<!-- A bounded flex frame; the ScrollArea fills it (flex:1 + min-height:0, which it owns) and
		     scrolls its overflowing content. The frame lives in the gallery's own markup so its height
		     is scoped here — the primitive itself stays unopinionated about its outer size. -->
		<div class="scroll-frame">
			<ScrollArea data-testid="ui-scrollarea">
				<Stack gap={2}>
					{#each Array.from({ length: 30 }, (_, i) => i) as i (i)}
						<div class="box">row {i}</div>
					{/each}
				</Stack>
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
				<Row gap={4} data-testid="ui-bar-end">
					<Button size="sm">Save</Button>
					<IconButton size="sm" label="Settings">⚙</IconButton>
				</Row>
			{/snippet}
		</Bar>
	</section>

	<section>
		<h2>Field composition (the north star)</h2>
		<!-- The proof: a Slider + NumberInput sharing ONE `cutoff` value inside a labelled Field, in
		     a handful of lines. Moving either drives the other; the label focuses the first control. -->
		<div class="form">
			<Field label="cutoff" hint="filter corner frequency" data-testid="ui-compose-field">
				<Row gap={4}>
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
				</Row>
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
				<Row gap={4} justify="end">
					<Button onclick={() => (dialogOpen = false)} data-testid="ui-dialog-cancel">Cancel</Button>
					<Button variant="primary" onclick={() => (dialogOpen = false)} data-testid="ui-dialog-confirm">
						Confirm
					</Button>
				</Row>
			</div>
		</Dialog>
	</section>

	<section>
		<h2>PanelShell (one header row)</h2>
		<!-- Title + toolbar + actions all compose into ONE header row (the Bar pusher pattern); the body
		     composes a ScrollArea so it scrolls independently. The e2e asserts exactly one header. -->
		<div class="shell-frame">
			<PanelShell title="Parameters" data-testid="ui-panelshell">
				{#snippet toolbar()}
					<Row gap={2} data-testid="ui-panelshell-toolbar">
						<Button size="sm">common</Button>
						<Button size="sm">advanced</Button>
					</Row>
				{/snippet}
				{#snippet actions()}
					<IconButton size="sm" label="Panel settings" data-testid="ui-panelshell-actions">⚙</IconButton>
				{/snippet}
				<ScrollArea data-testid="ui-panelshell-body">
					<Stack gap={2}>
						{#each Array.from({ length: 24 }, (_, i) => i) as i (i)}
							<div class="box">row {i}</div>
						{/each}
					</Stack>
				</ScrollArea>
			</PanelShell>
		</div>
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
		<Row gap={6}>
			{#each statusTones as tone (tone)}
				<Row gap={2}>
					<StatusDot {tone} data-testid={`ui-statusdot-${tone}`} />
					<span class="readout">{tone}</span>
				</Row>
			{/each}
		</Row>
		<Row gap={6}>
			{#each dotSizes as size (size)}
				<StatusDot tone="ok" {size} data-testid={`ui-statusdot-size-${size}`} />
			{/each}
		</Row>
	</section>

	<section>
		<h2>Spinner (CSS ring)</h2>
		<Row gap={6}>
			{#each spinnerSizes as size (size)}
				<Spinner {size} data-testid={`ui-spinner-${size}`} />
			{/each}
		</Row>
	</section>

	<section>
		<h2>EmptyState (centred placeholder)</h2>
		<!-- Full: icon + title + hint, centred on both axes inside a bounded frame. -->
		<div class="empty-frame">
			<EmptyState data-testid="ui-emptystate">
				{#snippet icon()}◎{/snippet}
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
	/* Gallery-local demo children — plain boxes so the layout primitives' spacing is what's measured. */
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
		font-family: var(--font-mono);
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
	.dialog-content h3 {
		margin: 0;
		font-size: var(--fs-strong);
		color: var(--text);
	}
	/* Bounded frame so the PanelShell's ScrollArea body has a height to scroll within. */
	.shell-frame {
		height: 14rem;
		width: 20rem;
		max-width: 100%;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		overflow: hidden;
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
