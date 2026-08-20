<script lang="ts">
	// The panel system's own stylesheet FIRST: it declares the `-default` tier of every token the
	// chrome reads, and `app.css` overrides that tier with this app's palette. Both are `:root`, so
	// the order in this file is the order the cascade sees.
	import 'tatami/tokens.css';
	import '../app.css';
	import '@xyflow/svelte/dist/style.css';
	import { onMount } from 'svelte';
	import { device } from '$lib/stores/device.svelte';

	let { children } = $props();

	onMount(() => device().init());
</script>

{@render children?.()}
