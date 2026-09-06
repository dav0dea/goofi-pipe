<!-- Selection inspector for one editor panel: parameters, metadata and errors for the selected
     node, on the side pane anchored to the host editor's edge. -->
<script lang="ts">
	import ParamForm from '$lib/inspector/ParamForm.svelte';
	import MetadataPanel from '$lib/editor/MetadataPanel.svelte';
	import SidePane from './SidePane.svelte';
	import { Button, ScrollArea } from '$lib/ui';
	import { graph } from '$lib/stores/graph.svelte';
	import type { NodeInstanceInfo } from '$lib/api/control';

	let {
		node,
		enabled,
		onClose
	}: {
		node: NodeInstanceInfo | null;
		enabled: boolean;
		/** Turn this editor's inspector off — the same switch the corner toggle flips. */
		onClose: () => void;
	} = $props();

	function restart(): void {
		if (!renderedNode) return;
		void graph()
			.restartNode(renderedNode.uid)
			.catch((e) => console.warn('restart failed', e));
	}

	function openEditor(): void {
		if (!renderedNode) return;
		void graph()
			.showNodeEditor(renderedNode.uid)
			.catch((e) => console.warn('editor failed', e));
	}

	/** Closing is a real outro, so the last node stays rendered until the slide finishes. */
	let renderedNode = $state<NodeInstanceInfo | null>(null);
	const open = $derived(enabled && node !== null);

	$effect(() => {
		if (open) renderedNode = node;
	});
</script>

<SidePane {open} onClosed={() => (renderedNode = null)} testid="auto-side-panel">
	<ScrollArea>
		<!-- Above the params: a plugin with sixty of them would bury it. -->
		{#if renderedNode?.editor}
			<section class="node-actions">
				<Button
					size="sm"
					onclick={openEditor}
					title="Open this plugin's own editor, in a window on the machine goofi runs on"
					data-testid="inspector-editor">▤ Open plugin editor</Button
				>
			</section>
		{/if}
		<ParamForm node={renderedNode} {onClose} />
		{#if renderedNode}
			<MetadataPanel node={renderedNode} />
			{#if renderedNode.error}
				<section class="node-error" data-testid="inspector-error">
					<div class="err-head">
						<header>Error</header>
						<Button
							variant="danger"
							size="sm"
							onclick={restart}
							title="Restart this node (respawn with the same params + links)"
							data-testid="inspector-restart">↻ Restart</Button
						>
					</div>
					<pre>{renderedNode.error}</pre>
				</section>
			{/if}
		{/if}
	</ScrollArea>
</SidePane>

<style>
	.node-actions {
		padding: var(--space-3) var(--space-6);
		border-bottom: 1px solid var(--border);
	}
	.node-error {
		padding: var(--space-6);
		border-top: 1px solid var(--border);
		background: var(--surface-1);
	}
	.err-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-5);
		margin-bottom: var(--space-3);
	}
	.node-error header {
		font-weight: 600;
		font-size: var(--fs-small);
		color: var(--danger);
	}
	/* Stated, not inherited: a bare <pre> takes app.css's `font: inherit`, which is the chrome face. */
	.node-error pre {
		font-family: var(--font-mono);
		font-size: var(--fs-micro);
		color: var(--text-dim);
		white-space: pre-wrap;
		word-break: break-word;
		margin: 0;
	}
</style>
