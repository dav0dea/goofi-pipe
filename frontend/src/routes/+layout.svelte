<script lang="ts">
	// Order is load-bearing: panelty declares the `-default` token tier, app.css overrides it.
	import 'panelty/tokens.css';
	import '../app.css';
	import '@xyflow/svelte/dist/style.css';
	import { onMount } from 'svelte';
	import { device } from '$lib/stores/device.svelte';

	let { children } = $props();

	onMount(() => device().init());
</script>

{@render children?.()}
