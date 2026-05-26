<script lang="ts">
	import type { DataFrame } from '$lib/codec/decode';

	type Props = { frame: DataFrame };
	const { frame }: Props = $props();

	const text = $derived(typeof frame.data === 'string' ? frame.data : String(frame.data));
</script>

<pre class="text" data-testid="string-viewer">{text}</pre>

<style>
	.text {
		width: 100%;
		height: 100%;
		min-height: 80px;
		font-family: var(--font-mono);
		font-size: 11px;
		color: var(--text);
		background: var(--bg-elev-1);
		margin: 0;
		padding: 6px 8px;
		border-radius: 4px;
		overflow: auto;
		white-space: pre-wrap;
		word-break: break-word;
	}
</style>
