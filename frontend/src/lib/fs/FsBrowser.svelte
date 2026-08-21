<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import { ui } from '$lib/stores/ui.svelte';
	import type { FsEntry, FsRoot } from '$lib/api/control';
	import { downloadPatch } from '$lib/api/patchFile';
	import { Bar, Button, Dialog, EmptyState, Icon, IconButton, ScrollArea, TextInput } from '$lib/ui';
	import { onMount, untrack } from 'svelte';

	type Props = {
		mode: 'save' | 'load';
		initialPath?: string | null;
		suggestedName?: string;
		onPick: (path: string) => void;
		onClose: () => void;
		/** The through-the-browser copy, for locations the backend cannot reach. */
		onFilePick: (file: File) => void;
	};
	const {
		mode,
		initialPath = null,
		suggestedName = '',
		onPick,
		onClose,
		onFilePick
	}: Props = $props();

	const g = graph();
	let cwd = $state('');
	let parent = $state<string | null>(null);
	let entries = $state<FsEntry[]>([]);
	let roots = $state<FsRoot[]>([]);
	let pathDraft = $state('');
	let filename = $state(untrack(() => suggestedName)); // initial draft; user edits own it
	let selected = $state<string | null>(null);
	let error = $state<string | null>(null);
	let pathBarEl = $state<HTMLDivElement | null>(null);
	let fileInput = $state<HTMLInputElement | null>(null);

	// A modal: the app's global chords stand down while it is up.
	const standdownId = $props.id();
	$effect(() => {
		ui().openEditor(standdownId);
		return () => ui().closeEditor(standdownId);
	});

	// A slower earlier listing must not clobber the directory the user has since navigated to.
	let navSeq = 0;

	async function go(path?: string | null): Promise<void> {
		const seq = ++navSeq;
		error = null;
		selected = null;
		try {
			const res = await g.listDir(path ?? undefined);
			if (seq !== navSeq) return;
			cwd = res.path;
			pathDraft = res.path;
			parent = res.parent;
			entries = res.entries;
			roots = res.roots;
		} catch (e) {
			if (seq !== navSeq) return;
			error = e instanceof Error ? e.message : String(e);
		}
	}

	// Adopt the typed path BEFORE navigating: `confirmSave` treats it as authoritative.
	function commitPath(path: string): void {
		pathDraft = path;
		void go(path);
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
		if (entry.kind === 'dir') return;
		if (entry.is_gfi) {
			selected = entry.path;
			if (mode === 'save') filename = entry.name.replace(/\.gfi$/, '');
		}
	}

	function confirmSave(): void {
		const name = filename.trim();
		if (!name) return;
		// The path bar is authoritative: a fast Save must not fall back to a stale cwd.
		const dir = (pathDraft || cwd).replace(/\/+$/, '');
		const full = `${dir}/${name.endsWith('.gfi') ? name : name + '.gfi'}`;
		onPick(full);
	}

	function confirmOpen(): void {
		if (selected) onPick(selected);
	}

	onMount(() => {
		// Focus after the first listing lands: earlier latches the input into editing mode.
		void go(initialPath).then(() => pathBarEl?.querySelector('input')?.focus());
	});

	const visible = $derived(entries.filter((e) => !e.hidden));
	const title = $derived(mode === 'save' ? 'Save patch' : 'Load patch');
</script>

<!-- `nokey`: SvelteFlow's delete key is a bare window listener, so Backspace here would delete the canvas selection. -->
<Dialog
	open
	class="nokey"
	{onClose}
	style="--dialog-pad: 0; --dialog-bg: var(--surface-1); --dialog-max-width: min(720px, 92vw); width: 100%"
	aria-label={title}
	data-testid="fs-browser"
>
	<div class="frame">
		<Bar style="--bar-pad-y: var(--space-2)">
			{#snippet start()}
				<span class="title">{title}</span>
			{/snippet}
			{#snippet end()}
				<IconButton variant="ghost" size="sm" label="Close" onclick={onClose}
					><Icon name="x" /></IconButton
				>
			{/snippet}
		</Bar>

		<div class="body">
			<nav class="roots">
				{#each roots as r (r.path)}
					<button class="root" class:active={cwd === r.path} onclick={() => go(r.path)}>{r.label}</button>
				{/each}
			</nav>

			<section class="files">
				<div class="pathbar" bind:this={pathBarEl}>
					<IconButton
						variant="ghost"
						size="sm"
						label="Up one level"
						disabled={!parent}
						onclick={() => go(parent)}>↑</IconButton
					>
					<TextInput
						inputmode="path"
						value={pathDraft}
						onChange={commitPath}
						autocomplete="off"
						data-testid="fs-path-input"
					/>
				</div>

				{#if error}
					<div class="err" data-testid="fs-error">{error}</div>
				{/if}

				<ScrollArea data-testid="fs-list">
					<ul class="rows">
						{#each visible as entry (entry.path)}
							<li>
								<button
									class="entry"
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
					</ul>
					{#if visible.length === 0}
						<EmptyState>
							{#snippet hint()}Empty folder.{/snippet}
						</EmptyState>
					{/if}
				</ScrollArea>
			</section>
		</div>

		<Bar class="fs-footer" style="--bar-wrap: wrap; --bar-pad-y: var(--space-2)">
			{#snippet start()}
				{#if mode === 'save'}
					<TextInput
						style="width: 14rem; flex: 0 1 auto; min-width: 0"
						value={filename}
						onChange={(v) => (filename = v)}
						placeholder="patch name"
						data-testid="fs-filename"
					/>
					<span class="ext">.gfi</span>
					<!-- A copy, not a save: it leaves the patch's remembered file alone. -->
					<Button variant="ghost" onclick={downloadPatch} data-testid="fs-download">
						Download a copy
					</Button>
				{:else}
					<Button variant="ghost" onclick={() => fileInput?.click()} data-testid="fs-upload">
						Open from this computer…
					</Button>
					<input
						bind:this={fileInput}
						type="file"
						accept=".gfi"
						hidden
						onchange={(e) => {
							const input = e.currentTarget;
							const file = input.files?.[0];
							// Cleared first: else picking the same file twice fires no second `change`.
							input.value = '';
							if (file) onFilePick(file);
						}}
					/>
				{/if}
			{/snippet}
			{#snippet end()}
				<Button variant="ghost" onclick={onClose}>Cancel</Button>
				{#if mode === 'save'}
					<Button variant="primary" onclick={confirmSave} data-testid="fs-save">Save</Button>
				{:else}
					<Button variant="primary" disabled={!selected} onclick={confirmOpen} data-testid="fs-open">
						Open
					</Button>
				{/if}
			{/snippet}
		</Bar>
	</div>
</Dialog>

<style>
	.frame {
		display: flex;
		flex-direction: column;
		min-width: 0;
		font-size: var(--fs-small);
		container: fs / inline-size;
	}
	.title {
		font-weight: 600;
	}
	.body {
		display: flex;
		min-height: 0;
		/* `dvh`, not `vh`: on a phone `vh` is the largest viewport, so the modal would overflow. */
		height: min(24rem, 55dvh);
	}
	.roots {
		flex: 0 0 8.75rem;
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
		padding: var(--space-4) var(--space-3);
		background: var(--surface-2);
		overflow-y: auto;
	}
	.root {
		font: inherit;
		background: transparent;
		border: none;
		color: var(--text);
		text-align: left;
		padding: var(--space-3) var(--space-4);
		border-radius: var(--radius-sm);
		cursor: pointer;
		transition: background var(--dur-fast) var(--ease);
	}
	.root.active,
	.root:hover {
		background: var(--accent-fill);
	}
	.files {
		flex: 1;
		display: flex;
		flex-direction: column;
		min-width: 0;
	}
	/* The mono face is stated on the strip because `TextInput` is `font: inherit`. */
	.pathbar {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		padding: var(--space-4);
		font-family: var(--font-mono);
	}
	.err {
		color: var(--warning);
		padding: var(--space-3) var(--space-5);
		font-family: var(--font-mono);
	}
	.rows {
		list-style: none;
		margin: 0;
		padding: var(--space-2) 0;
	}
	.entry {
		font: inherit;
		font-family: var(--font-mono);
		display: flex;
		align-items: center;
		gap: var(--space-4);
		width: 100%;
		text-align: left;
		background: transparent;
		border: none;
		border-radius: var(--radius-sm);
		color: var(--text);
		padding: var(--space-2) var(--space-6);
		cursor: pointer;
		transition: background var(--dur-fast) var(--ease);
	}
	.entry:hover {
		background: color-mix(in srgb, var(--accent) 8%, transparent);
	}
	.entry.sel {
		background: var(--accent-fill);
	}
	.entry .ico {
		flex: 0 0 auto;
		width: 1rem;
		text-align: center;
	}
	.entry.gfi .ico {
		color: var(--accent);
	}
	.entry .nm {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	/* `:global` because the class travels to `Bar` as a prop; `.frame` keeps it scoped. */
	.frame :global(.fs-footer) {
		font-family: var(--font-mono);
	}
	.ext {
		color: var(--text-muted);
	}
	@container fs (max-width: 30rem) {
		.body {
			flex-direction: column;
		}
		.roots {
			flex: 0 0 auto;
			flex-direction: row;
			overflow-x: auto;
			overflow-y: hidden;
		}
		.root {
			flex: 0 0 auto;
			white-space: nowrap;
		}
	}
</style>
