<!--
  Bar — a horizontal toolbar / header strip (spec §2.3), the "pusher pattern": a `start` snippet
  group hugs the left, a flex spacer eats the slack, and an `end` snippet group is pushed to the
  right. Either group is optional (the spacer alone still pushes a lone `end` right / keeps a lone
  `start` left). Merged `class`, forwarded `data-testid` (and any other attribute) via `...rest`.

  Padding, height, gap and chrome are all F tokens, each exposed as a `var(--bar-*, <token>)` hook
  so a consumer can retune one instance from a wrapper — the spec §1 per-instance-theming mechanism.
  Defaults give the panel-header look: a `--surface-2` strip one step above the `--surface-1` body
  it sits on, which separates it WITHOUT a hairline (D5), at a `--panel-header-h` min-height that
  grows to the touch floor under a coarse pointer.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import type { HTMLAttributes } from 'svelte/elements';

	let {
		start,
		end,
		class: klass = '',
		...rest
	}: HTMLAttributes<HTMLDivElement> & {
		/** Left-hugging group. */
		start?: Snippet;
		/** Right-pushed group. */
		end?: Snippet;
	} = $props();
</script>

<div {...rest} class={`ui-bar ${klass}`.trim()}>
	{#if start}
		<div class="ui-bar-group">{@render start()}</div>
	{/if}
	<div class="ui-bar-spacer"></div>
	{#if end}
		<div class="ui-bar-group">{@render end()}</div>
	{/if}
</div>

<style>
	.ui-bar {
		display: flex;
		align-items: center;
		/* Opt-in wrapping (default off, so every existing bar is byte-identical). A bar whose two
		   groups have a real minimum — the file browser's footer is a filename plus Cancel/Save —
		   has to put the second group on a second line rather than push it off the edge, and
		   `wrap` does that at the width it actually stops fitting rather than at a device class. */
		flex-wrap: var(--bar-wrap, nowrap);
		min-width: 0;
		min-height: var(--bar-height, var(--panel-header-h));
		padding: var(--bar-pad-y, var(--space-2)) var(--bar-pad-x, var(--space-4));
		gap: var(--bar-gap, var(--space-4));
		background: var(--bar-bg, var(--surface-2));
		border-bottom: var(--bar-border, none);
	}
	/* Each group lays its own items in a row and can shrink (so a long label ellipsis's, not overflows). */
	.ui-bar-group {
		display: flex;
		align-items: center;
		gap: var(--bar-gap, var(--space-4));
		min-width: 0;
	}
	/* The `end` group stays right-aligned even once it has wrapped onto its own line, where the
	   spacer above it is no longer between the two. A no-op on a single line (the spacer already
	   ate the slack), so nothing moves in the default, non-wrapping bar. */
	.ui-bar-group:last-child {
		margin-left: auto;
	}
	/* The pusher: eats the slack between start and end, forcing end to the right edge. */
	.ui-bar-spacer {
		flex: 1 1 auto;
	}
</style>
