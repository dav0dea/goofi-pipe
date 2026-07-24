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
		type ButtonVariant,
		type ButtonSize
	} from '$lib/ui';

	const variants: ButtonVariant[] = ['default', 'primary', 'ghost', 'danger'];
	const sizes: ButtonSize[] = ['sm', 'md'];
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
	let refreshValue = $state('sine');
	let refreshing = $state(false);
	let refreshCount = $state(0);
	let textText = $state('hello');
	let textDecimal = $state('3.14');
	let textSearch = $state('');
	let textPath = $state('/home/user/patch.gfi');
	let triggerCount = $state(0);
	let toggled = $state(false);

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
</style>
