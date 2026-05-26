<script lang="ts">
	import '../app.css';
	import '@xyflow/svelte/dist/style.css';

	let { children } = $props();
</script>

{@render children?.()}
