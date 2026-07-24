<!--
  /dev/ui — the primitive gallery harness (spec §6). A static, backend-free showcase of
  every `$lib/ui` primitive in its variants; the committed `tests/e2e` gallery specs drive
  it as the "failing test first" for a UI lib that has no vitest mount runner. Later P tasks
  extend this route + its specs. Each sample carries a distinct `data-testid`.
-->
<script lang="ts">
	import { Button, IconButton, type ButtonVariant, type ButtonSize } from '$lib/ui';

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
</style>
