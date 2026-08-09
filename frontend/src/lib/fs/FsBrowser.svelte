<!--
  Backend filesystem browser modal. Two modes: 'save' (pick a directory + type a
  filename) and 'load' (pick an existing .gfi). Full-FS, no jail (trusted LAN).

  The modal shell is the `Dialog` primitive: it owns the backdrop, Escape, the focus
  trap and the native top layer, so this component keeps no overlay, no window key
  handler and no z-index of its own. The nav rows (`.root` / `.entry`) stay bespoke —
  they are list rows, not actions — while every real control is a ui primitive.
-->
<script lang="ts">
	import { graph } from '$lib/stores/graph.svelte';
	import { ui } from '$lib/stores/ui.svelte';
	import type { FsEntry, FsRoot } from '$lib/api/control';
	import { Bar, Button, Dialog, EmptyState, Icon, IconButton, ScrollArea, TextInput } from '$lib/ui';
	import { onMount, untrack } from 'svelte';

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
	let pathBarEl = $state<HTMLDivElement | null>(null);

	// This browser is a MODAL: while it is up the app's global chords stand down, so a Ctrl+Z on a
	// focused file row can't undo a graph command behind it (and Ctrl+S can't re-enter Save). Held in
	// the ui store's ref-counted editor set — the same standdown the inspector's fx editor uses — and
	// released by the effect's cleanup when AppShell unmounts the browser.
	const standdownId = $props.id();
	$effect(() => {
		ui().openEditor(standdownId);
		return () => ui().closeEditor(standdownId);
	});

	// Monotonic navigation token: a slower earlier listing must not clobber the directory the user
	// has since navigated to (open the browser, immediately type a path → the in-flight initial
	// listDir used to land last and bounce them home).
	let navSeq = 0;

	async function go(path?: string | null): Promise<void> {
		const seq = ++navSeq;
		error = null;
		selected = null;
		try {
			const res = await g.listDir(path ?? undefined);
			if (seq !== navSeq) return; // superseded by a newer navigation
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

	// Adopt the typed path BEFORE navigating: it is what `confirmSave` treats as authoritative, and
	// the listing is a round trip away. A failed `go()` leaves it standing, so the user can correct it.
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

	onMount(() => {
		// Focus the path bar once the first listing has landed: focusing earlier latches the input's
		// live-value into editing mode, which would suppress that very echo. `showModal()` has already
		// moved focus into the dialog — this only puts it where typing is useful.
		void go(initialPath).then(() => pathBarEl?.querySelector('input')?.focus());
	});

	const visible = $derived(entries.filter((e) => !e.hidden));
	const title = $derived(mode === 'save' ? 'Save patch' : 'Load patch');
</script>

<!-- `nokey`: the node editor delegates deletion OUT of its own guarded keydown to SvelteFlow
     (`deleteKey` + `ondelete`), whose `KeyHandler` is a bare window listener filtered only by
     "is the target a text field" — so neither of the editor's two standdowns (my panel is active,
     the press did not come from inside a `dialog[open]`) can reach it. Backspace is what a file
     browser trains you to press for "up a folder", and behind this modal it deleted the canvas
     selection, unreversibly in place (Ctrl+Z is stood down while a dialog is up). xyflow honours
     `closest('.nokey')`, so the standdown goes on the modal that took the keyboard rather than as
     a third condition on the editor. This is the app's only real modal. -->
<Dialog
	open
	class="nokey"
	{onClose}
	style="--dialog-pad: 0; --dialog-bg: var(--surface-1); --dialog-max-width: min(720px, 92vw); width: 100%"
	aria-label={title}
	data-testid="fs-browser"
>
	<div class="frame">
		<Bar>
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

		<!-- The one bar in the app whose two groups both have a real minimum — a filename field and
		     Cancel/Save. It wraps at the width they stop fitting; every other bar keeps `nowrap`.
		     The WRAP is what makes the footer fit a 320px phone; the field's 14rem is DESKTOP
		     geometry and is stated as a width, not as a flex basis. `flex: 1 1 8rem` cost the
		     desktop ~40px for nothing: the grow can never fire (`.ui-bar-group` is `0 1 auto` and
		     `.ui-bar-spacer` owns the bar's slack), and a shrinkable item's max-content
		     CONTRIBUTION collapses toward its own intrinsic size — an `<input>`'s default `size=20`
		     — so the basis was not what the group asked for either. A definite width is.
		     `flex: 0 1` + `min-width: 0` keep it shrinkable on the narrow line it wraps onto. -->
		<Bar class="fs-footer" style="--bar-wrap: wrap">
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
				{:else if onUpload}
					<Button variant="ghost" onclick={onUpload} data-testid="fs-upload">
						Upload from this computer…
					</Button>
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
		/* The modal is 92vw, so its own width — not the viewport's, and not a device class — is what
		   decides whether a sidebar still fits beside the list (D-R6). */
		container: fs / inline-size;
	}
	.title {
		font-weight: 600;
	}
	.body {
		display: flex;
		min-height: 0;
		/* The list is the only scroller, and a fixed body height keeps the modal from resizing as
		   the user walks directories of different lengths. `dvh`, not `vh`: on a phone `vh` is the
		   LARGEST viewport (browser chrome retracted), so a `vh`-sized modal overflows the screen
		   for as long as the address bar is showing. */
		height: min(24rem, 55dvh);
	}
	/* Root shortcuts read as a sidebar off the same surface step as the bars — no divider needed. */
	.roots {
		flex: 0 0 8.75rem;
		display: flex;
		flex-direction: column;
		gap: var(--space-1);
		padding: var(--space-4) var(--space-3);
		background: var(--surface-2);
		overflow-y: auto;
	}
	/* `.root` / `.entry` are list rows, not actions, so they stay bespoke — which means stating the
	   whole of their appearance. Both wash accent on hover/selection, so the radius that rounds that
	   wash and the fade that eases it in are theirs to declare: they came from app.css's base
	   `button` skin until M-Task 7 stripped it (only the `font: inherit` reset survives there). */
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
	/* A path is DATA (D-T3) — the same string the rows below it hold, one directory further down —
	   so the bar that carries it reads in mono. Stated HERE, on the strip, because `TextInput` is
	   `font: inherit` by design: the seam that encloses the control is the only place a consumer can
	   hand it a face. The ↑ beside it inherits the same mono, which is what the file rows do too. */
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
	/* The scroller clips horizontally, so a long name ellipsis's instead of being cut mid-glyph. */
	.entry .nm {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	/* The footer's own seam, and the same call as `.pathbar`: the name a patch is saved under is the
	   file the list above will show, so it is data. The face has to be stated on the strip because
	   the field inside it is `font: inherit` — and nothing else in the bar moves, since `Button`
	   declares its sans itself. `:global`, because the class travels to `Bar` as a prop (the idiom
	   `ParamForm` uses for `.pf-identity-bar`); `.frame` keeps it scoped to this component's tree. */
	.frame :global(.fs-footer) {
		font-family: var(--font-mono);
	}
	.ext {
		color: var(--text-muted);
	}
	/* Below the width where a fixed sidebar leaves a usable file list, the roots lie DOWN: a
	   horizontal strip above the list instead of a column beside it. Same rows, same order, one less
	   axis — a different representation of one state, not a second layout (D-R2). */
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
