<!--
  /dev/ui — the primitive gallery harness (spec §6). A static, backend-free showcase of
  every `$lib/ui` primitive in its variants; the committed `tests/e2e` gallery specs drive
  it as the "failing test first" for a UI lib that has no vitest mount runner. Later P tasks
  extend this route + its specs. Each sample carries a distinct `data-testid`.
-->
<script lang="ts">
	import { Button, IconButton, Stack, Row, ScrollArea, Bar, type ButtonVariant, type ButtonSize } from '$lib/ui';

	const variants: ButtonVariant[] = ['default', 'primary', 'ghost', 'danger'];
	const sizes: ButtonSize[] = ['sm', 'md'];
	// Single-character glyphs so the IconButton glyph stays visibly small.
	const glyphs: Record<ButtonVariant, string> = {
		default: '⚙', // gear
		primary: '+',
		ghost: '⟳', // refresh
		danger: '×' // multiplication sign
	};
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
</style>
