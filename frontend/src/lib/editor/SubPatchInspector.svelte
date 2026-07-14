<!--
  Inspector body for a selected sub-patch (virtual) node. Stands in for the
  param groups in ParamPanel. Shows the sharing state, the structural actions
  that used to live on the node body (duplicate-as-shared / make-unique /
  expand), and — for a shared sub-patch — the list of sibling instances it is
  strict-mirrored with (click to jump to one).
-->
<script lang="ts">
	import type { NodeInstanceInfo } from '$lib/api/control';
	import { graph } from '$lib/stores/graph.svelte';
	import { selection } from '$lib/stores/selection.svelte';

	let { node }: { node: NodeInstanceInfo } = $props();

	const g = graph();
	const sel = selection();

	const instId = $derived(node.subpatch?.instId ?? node.name);
	// Read live instance state so the panel reflects make-unique/duplicate
	// immediately (the synth node is rebuilt on the next selection recompute).
	const inst = $derived(g.instances[instId] ?? null);
	const shared = $derived(Boolean(inst?.def_id));
	const memberCount = $derived(Object.keys(inst?.members ?? {}).length);
	const siblings = $derived(inst?.siblings ?? []);

	function duplicateShared(): void {
		const p = inst?.pos;
		const pos: [number, number] = p ? [p[0] + 60, p[1] + 60] : [0, 0];
		void g.duplicateShared(instId, pos).catch((e) => console.warn('duplicate shared failed', e));
	}
	function makeUnique(): void {
		void g.makeUnique(instId).catch((e) => console.warn('make unique failed', e));
	}
	function expand(): void {
		void g.expandInstance(instId).catch((e) => console.warn('expand failed', e));
	}
	function focusSibling(id: string): void {
		const panel = sel.activeEditorId;
		if (panel) sel.selectNodes(panel, [id]);
	}
</script>

<section class="panel" data-testid="subpatch-inspector">
	<header>
		<span class="glyph" class:shared>{shared ? '⇄' : '▣'}</span>
		<div class="titles">
			<div class="title">{shared ? 'Shared sub-patch' : 'Sub-patch'}</div>
			<div class="sub">{inst?.name ?? instId}</div>
		</div>
		<span class="badge">{memberCount} member{memberCount === 1 ? '' : 's'}</span>
	</header>

	<div class="rows">
		<div class="sharing">
			<span class="state">{shared ? 'Mirrored (strict)' : 'Independent'}</span>
			{#if shared}
				<button class="act" data-testid="subpatch-make-unique" onclick={makeUnique}>Make unique</button>
			{:else}
				<button class="act" data-testid="subpatch-share" onclick={duplicateShared}
					>Duplicate as shared ⇄</button
				>
			{/if}
		</div>

		{#if shared}
			<div class="mirror">
				<div class="mirror-label">Mirrored with</div>
				{#if siblings.length === 0}
					<div class="mirror-empty">no other instances</div>
				{:else}
					<ul class="mirror-list" data-testid="subpatch-siblings">
						{#each siblings as sib (sib)}
							{@const sibName = g.instances[sib]?.name ?? sib}
							<li>
								<button class="sib" onclick={() => focusSibling(sib)} title="Select {sibName}"
									>⇄ {sibName}</button
								>
							</li>
						{/each}
					</ul>
				{/if}
			</div>
		{/if}

		<button class="act ghost" data-testid="subpatch-expand-inspector" onclick={expand}
			>Expand (dissolve into nodes)</button
		>
	</div>
</section>

<style>
	.panel {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	header {
		display: flex;
		gap: 10px;
		align-items: center;
		padding: 10px 12px;
		border-bottom: 1px solid var(--border);
		background: var(--bg-elev-1);
	}
	.glyph {
		font-size: 14px;
		color: var(--accent);
	}
	.glyph.shared {
		color: var(--cat-array, var(--accent));
	}
	.titles {
		min-width: 0;
		flex: 1;
	}
	.title {
		font-size: 14px;
		font-weight: 600;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.sub {
		color: var(--text-faint);
		font-family: var(--font-mono);
		font-size: 10px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.badge {
		font-family: var(--font-mono);
		font-size: 9px;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		padding: 2px 6px;
		border-radius: 999px;
		color: var(--text-dim);
		background: color-mix(in srgb, var(--accent) 14%, transparent);
		border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent);
		flex-shrink: 0;
	}
	.rows {
		display: flex;
		flex-direction: column;
		gap: 12px;
		padding: 12px;
	}
	.sharing {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
	}
	.state {
		font-family: var(--font-mono);
		font-size: 11px;
		color: var(--text-dim);
	}
	.act {
		background: transparent;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		cursor: pointer;
		font-size: 11px;
		padding: 5px 9px;
		white-space: nowrap;
	}
	.act:hover {
		border-color: var(--accent);
	}
	.act.ghost {
		color: var(--text-dim);
		justify-content: center;
	}
	.mirror {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.mirror-label {
		font-size: 10px;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--text-faint);
	}
	.mirror-empty {
		font-size: 11px;
		color: var(--text-faint);
	}
	.mirror-list {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.sib {
		width: 100%;
		text-align: left;
		background: var(--bg-elev-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-dim);
		cursor: pointer;
		font-family: var(--font-mono);
		font-size: 11px;
		padding: 4px 8px;
	}
	.sib:hover {
		color: var(--text);
		border-color: var(--accent);
	}
</style>
