<!--
  Backend filesystem browser modal. Two modes: 'save' (pick a directory + type a
  filename) and 'load' (pick an existing .gfi). Full-FS, no jail (trusted LAN).
  Renders in the top-level modal band above all other chrome.
-->
<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import type { FsEntry, FsRoot } from '$lib/api/control';
	import { onMount, tick, untrack } from 'svelte';

	type Props = {
		mode: 'save' | 'load';
		initialPath?: string | null;
		suggestedName?: string;
		onPick: (path: string) => void;
		onClose: () => void;
		onUpload?: () => void;
	};
	const { mode, initialPath = null, suggestedName = '', onPick, onClose, onUpload }: Props = $props();

	const g = graph();
	let cwd = $state('');
	let parent = $state<string | null>(null);
	let entries = $state<FsEntry[]>([]);
	let roots = $state<FsRoot[]>([]);
	let pathDraft = $state('');
	let filename = $state(untrack(() => suggestedName)); // initial draft; user edits own it
	let selected = $state<string | null>(null);
	let error = $state<string | null>(null);
	let firstInput = $state<HTMLInputElement | null>(null);

	async function go(path?: string | null): Promise<void> {
		error = null;
		selected = null;
		try {
			const res = await g.listDir(path ?? undefined);
			cwd = res.path;
			pathDraft = res.path;
			parent = res.parent;
			entries = res.entries;
			roots = res.roots;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}

	function openEntry(entry: FsEntry): void {
		if (entry.kind === 'dir') {
			void go(entry.path);
		} else if (entry.is_gfi) {
			if (mode === 'load') onPick(entry.path);
			else filename = entry.name.replace(/\.gfi$/, '');
		}
	}

	function clickEntry(entry: FsEntry): void {
		if (entry.kind === 'dir') return; // single click on a dir just highlights
		if (entry.is_gfi) {
			selected = entry.path;
			if (mode === 'save') filename = entry.name.replace(/\.gfi$/, '');
		}
	}

	function confirmSave(): void {
		const name = filename.trim();
		if (!name) return;
		// The path bar is authoritative for the target directory — it holds the
		// user's typed/navigated path even if a listDir() is still in flight, so
		// a fast Save click can't fall back to a stale (or empty) cwd.
		const dir = (pathDraft || cwd).replace(/\/+$/, '');
		const full = `${dir}/${name.endsWith('.gfi') ? name : name + '.gfi'}`;
		onPick(full);
	}

	function confirmOpen(): void {
		if (selected) onPick(selected);
	}

	function onKeydown(e: KeyboardEvent): void {
		if (e.key === 'Escape') {
			e.preventDefault();
			onClose();
		}
	}

	onMount(() => {
		void go(initialPath);
		tick().then(() => firstInput?.focus());
	});

	const visible = $derived(entries.filter((e) => !e.hidden));
</script>

<svelte:window onkeydown={onKeydown} />

<div class="fs-backdrop" onclick={onClose} role="presentation"></div>
<div
	class="fs-modal"
	role="dialog"
	aria-label={mode === 'save' ? 'Save patch' : 'Load patch'}
	data-testid="fs-browser"
>
	<header>
		<span class="title">{mode === 'save' ? 'Save patch' : 'Load patch'}</span>
		<button class="x" onclick={onClose} aria-label="Close">✕</button>
	</header>

	<div class="body">
		<nav class="roots">
			{#each roots as r (r.path)}
				<button class="root" class:active={cwd === r.path} onclick={() => go(r.path)}>{r.label}</button>
			{/each}
		</nav>

		<section class="files">
			<div class="pathbar">
				<button class="up" disabled={!parent} onclick={() => go(parent)} title="Up one level">↑</button>
				<input
					bind:this={firstInput}
					bind:value={pathDraft}
					onkeydown={(e) => {
						if (e.key === 'Enter') void go(pathDraft);
					}}
					spellcheck="false"
					autocomplete="off"
					data-testid="fs-path-input"
				/>
			</div>

			{#if error}
				<div class="err" data-testid="fs-error">{error}</div>
			{/if}

			<ul class="list" data-testid="fs-list">
				{#each visible as entry (entry.path)}
					<li>
						<button
							class="entry"
							class:dir={entry.kind === 'dir'}
							class:gfi={entry.is_gfi}
							class:sel={selected === entry.path}
							onclick={() => clickEntry(entry)}
							ondblclick={() => openEntry(entry)}
							data-testid="fs-entry"
						>
							<span class="ico">{entry.kind === 'dir' ? '📁' : entry.is_gfi ? '◆' : '·'}</span>
							<span class="nm">{entry.name}</span>
						</button>
					</li>
				{/each}
				{#if visible.length === 0}
					<li class="empty">Empty folder.</li>
				{/if}
			</ul>
		</section>
	</div>

	<footer>
		{#if mode === 'save'}
			<input class="fname" bind:value={filename} placeholder="patch name" data-testid="fs-filename" />
			<span class="ext">.gfi</span>
			<div class="spacer"></div>
			<button class="ghost" onclick={onClose}>Cancel</button>
			<button class="primary" onclick={confirmSave} data-testid="fs-save">Save</button>
		{:else}
			{#if onUpload}
				<button class="ghost" onclick={onUpload} data-testid="fs-upload">Upload from this computer…</button>
			{/if}
			<div class="spacer"></div>
			<button class="ghost" onclick={onClose}>Cancel</button>
			<button class="primary" disabled={!selected} onclick={confirmOpen} data-testid="fs-open">Open</button>
		{/if}
	</footer>
</div>

<style>
	.fs-backdrop {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.45);
		z-index: 1000;
	}
	.fs-modal {
		position: fixed;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		width: min(720px, 92vw);
		height: min(560px, 86vh);
		display: flex;
		flex-direction: column;
		background: var(--surface-1);
		border: 1px solid var(--border-strong);
		border-radius: var(--radius-md);
		box-shadow: var(--shadow-2);
		z-index: 1001;
		font-size: 12px;
		overflow: hidden;
	}
	header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 10px 12px;
		border-bottom: 1px solid var(--border);
		font-weight: 600;
	}
	header .x {
		background: transparent;
		border: none;
		color: var(--text-dim);
		cursor: pointer;
		font-size: 13px;
	}
	.body {
		flex: 1;
		display: flex;
		min-height: 0;
	}
	.roots {
		flex: 0 0 140px;
		border-right: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		padding: 8px 6px;
		gap: 2px;
	}
	.root {
		background: transparent;
		border: none;
		color: var(--text);
		text-align: left;
		padding: 6px 8px;
		border-radius: var(--radius-sm);
		cursor: pointer;
	}
	.root.active,
	.root:hover {
		background: color-mix(in srgb, var(--accent) 14%, transparent);
	}
	.files {
		flex: 1;
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	.pathbar {
		display: flex;
		gap: 6px;
		padding: 8px;
		border-bottom: 1px solid var(--border);
	}
	.pathbar .up {
		flex: 0 0 auto;
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		cursor: pointer;
		padding: 0 8px;
	}
	.pathbar input {
		flex: 1;
		min-width: 0;
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		padding: 4px 8px;
		font-family: var(--font-mono);
	}
	.err {
		color: var(--warning);
		padding: 6px 10px;
		font-family: var(--font-mono);
	}
	.list {
		flex: 1;
		overflow-y: auto;
		list-style: none;
		margin: 0;
		padding: 4px 0;
	}
	.entry {
		display: flex;
		align-items: center;
		gap: 8px;
		width: 100%;
		text-align: left;
		background: transparent;
		border: none;
		color: var(--text);
		padding: 5px 12px;
		cursor: pointer;
		font-family: var(--font-mono);
	}
	.entry:hover {
		background: color-mix(in srgb, var(--accent) 8%, transparent);
	}
	.entry.sel {
		background: color-mix(in srgb, var(--accent) 18%, transparent);
	}
	.entry .ico {
		flex: 0 0 auto;
		width: 14px;
		text-align: center;
	}
	.entry.gfi .ico {
		color: var(--accent);
	}
	.empty {
		color: var(--text-muted);
		padding: 12px;
		text-align: center;
		list-style: none;
	}
	footer {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 10px 12px;
		border-top: 1px solid var(--border);
	}
	footer .spacer {
		flex: 1;
	}
	.fname {
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		padding: 4px 8px;
		font-family: var(--font-mono);
	}
	.ext {
		color: var(--text-muted);
		font-family: var(--font-mono);
	}
	button.ghost,
	button.primary {
		border-radius: var(--radius-sm);
		padding: 5px 12px;
		cursor: pointer;
		font-size: 12px;
	}
	button.ghost {
		background: transparent;
		border: 1px solid var(--border);
		color: var(--text);
	}
	button.primary {
		background: var(--accent);
		border: 1px solid var(--accent);
		color: var(--surface-1);
	}
	button.primary:disabled {
		opacity: 0.5;
		cursor: default;
	}
</style>
