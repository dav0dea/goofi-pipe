<script lang="ts">
	import '../app.css';
	import '@xyflow/svelte/dist/style.css';
	import { onMount } from 'svelte';
	import { device } from '$lib/stores/device.svelte';

	let { children } = $props();

	onMount(() => device().init());
</script>

{@render children?.()}
