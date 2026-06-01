<!-- The framework's default/fallback content: an empty panel inviting the user
     to pick a content type from the header dropdown. -->
<script lang="ts">
	import type { PanelProps } from './registry';
	// Conforms to the panel contract; this panel needs none of the props.
	// eslint-disable-next-line @typescript-eslint/no-unused-vars
	let {}: PanelProps = $props();
</script>

<div class="empty">
	<div class="hint">
		<span class="big">Empty panel</span>
		<span class="sub">Pick content from the <strong>▾</strong> menu above</span>
	</div>
</div>

<style>
	.empty {
		display: grid;
		place-items: center;
		height: 100%;
		background: var(--bg);
	}
	.hint {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 6px;
		color: var(--text-faint);
		text-align: center;
	}
	.big {
		font-size: 0.95rem;
	}
	.sub {
		font-size: 0.8rem;
	}
</style>
