<!--
  Agent panel — a terminal on an agent harness running in this patch's workspace.

  The `Terminal` is NOT this component's (`$lib/stores/termSession`): it is kept per instance so
  closing this panel and opening another loses nothing, and this component's whole job on mount is
  `attach(el)` — one `term.open` into a fresh element, plus a size nudge so a full-screen TUI
  repaints itself. Nothing is replayed, because there is nothing to replay from: the manager keeps
  no grid.

  Which instance the panel shows is client-local viewpoint (see the store's note): it never reaches
  the manager, so it cannot dirty the patch or enter undo.

  Closing this view is not killing the agent, so the ✕ asks — `register.ts` routes the panel
  header's close into the store, which raises the question. The DIALOG is the shell's
  (`app/AgentClose.svelte`): it is about an instance, not a view, and a question that needed a panel
  made the TopBar's badge open one — an authoring write, for asking.
-->
<script lang="ts">
	import { Terminal } from '@xterm/xterm';
	import { FitAddon } from '@xterm/addon-fit';
	import '@xterm/xterm/css/xterm.css';
	import type { PanelProps } from '$lib/workspace/registry';
	import { harnesses, harnessLabel } from '$lib/stores/harness.svelte';
	import { termSession, type TerminalLike } from '$lib/stores/termSession';
	import { Bar, Button, EmptyState, Select, StatusDot } from '$lib/ui';
	import { untrack } from 'svelte';

	let { panelId }: PanelProps = $props();
	const hs = harnesses();

	// Mounted panels are what the store hands instances out against, so this registration is the
	// panel's whole lifecycle. `untrack`: claiming WRITES the map it reads, and a tracked effect
	// would then re-run — unmounting and remounting on every roster event, which would re-claim
	// the instance a Detach just let go of.
	$effect(() => {
		untrack(() => hs.mount(panelId));
		return () => hs.unmount(panelId);
	});

	const id = $derived(hs.instanceFor(panelId));
	const inst = $derived(hs.instances.find((i) => i.id === id) ?? null);

	let host = $state<HTMLDivElement | null>(null);

	const token = (n: string): string =>
		getComputedStyle(document.documentElement).getPropertyValue(n).trim();

	/** xterm and its fit addon as ONE object, because the addon has to live with the terminal that
	 * outlives this panel — a component holding its own would have nothing to measure with after a
	 * remount. Colours and face come from the app's tokens, so `app.css` stays the SSOT for a
	 * surface that draws itself on a canvas. */
	function makeTerminal(): TerminalLike {
		const t = new Terminal({
			fontFamily: token('--font-mono'),
			fontSize: 12,
			scrollback: 5000,
			theme: {
				background: token('--surface-1'),
				foreground: token('--text'),
				cursor: token('--accent')
			}
		});
		const fit = new FitAddon();
		t.loadAddon(fit);
		return Object.assign(t, { proposeDimensions: () => fit.proposeDimensions() });
	}

	$effect(() => {
		const el = host;
		const at = id;
		if (!el || !at) return;
		const s = termSession(at, makeTerminal);
		s.attach(el);
		// The ONLY thing that proposes a size. An inbound authoritative size sets the terminal
		// directly (the store's invariant) — feeding one back through the fit addon would put two
		// views of the same instance in a loop. The observer also covers the soft keyboard: the host
		// pads by `--kb-inset`, so its content box moves when the keyboard does.
		const ro = new ResizeObserver(() => s.refit());
		ro.observe(el);
		return () => {
			ro.disconnect();
			// Unmounting gives up the SIZE, not the stream: the socket stays open so the transcript
			// keeps arriving for whoever shows this instance next.
			s.retract();
		};
	});
</script>

<div class="agent">
	<Bar>
		{#snippet start()}
			{#if inst}
				<StatusDot
					tone={inst.state === 'running' ? 'ok' : inst.state === 'stopping' ? 'warn' : 'error'}
					aria-label={inst.state}
				/>
				<!-- The switcher: one panel, any instance. Two agent panels can already show two, but
				     changing which one this panel watches must not cost a second panel. -->
				<Select
					value={inst.id}
					options={hs.instances.map((i) => i.id)}
					labels={Object.fromEntries(hs.instances.map((i) => [i.id, harnessLabel(i)]))}
					onChange={(v) => hs.show(panelId, v)}
					data-testid="agent-switcher"
				/>
			{/if}
		{/snippet}
		{#snippet end()}
			{#if inst}
				<Button
					size="sm"
					variant="ghost"
					data-testid="agent-close"
					onclick={() => hs.requestClose(inst.id)}
				>
					{inst.state === 'exited' ? `Dismiss (exit ${inst.exit_code ?? '?'})` : 'Close…'}
				</Button>
			{/if}
		{/snippet}
	</Bar>
	{#if id}
		<div class="host" bind:this={host} data-testid="agent-terminal"></div>
	{:else}
		<EmptyState data-testid="agent-launcher">
			{#snippet title()}No agent running here{/snippet}
			{#snippet hint()}
				{hs.detected.length
					? 'It runs in this patch workspace, editing the patch with you.'
					: 'No agent harness found on PATH.'}
			{/snippet}
			<div class="choices">
				{#each hs.instances as i (i.id)}
					<Button size="sm" onclick={() => hs.show(panelId, i.id)}>Attach {harnessLabel(i)}</Button>
				{/each}
				{#each hs.detected as d (d.harness)}
					<Button
						size="sm"
						variant="primary"
						title={d.version ?? d.path}
						data-testid={`agent-launch-${d.harness}`}
						onclick={() => void hs.launch(panelId, d.harness)}>Launch {d.harness}</Button
					>
				{/each}
			</div>
		</EmptyState>
	{/if}
</div>

<style>
	.agent {
		display: flex;
		flex-direction: column;
		height: 100%;
		min-height: 0;
		background: var(--surface-1);
	}
	/* The terminal's box is the PANEL's, never its content's: xterm lays itself out against this
	   element, so a host that grew with the screen it hosts would re-measure forever. `overflow`
	   is what letterboxes — a view that is not the size owner draws a screen wider or taller than
	   its panel, and clips rather than reflowing someone else's terminal.
	   `--kb-inset` is the soft keyboard's overlap (published on <html> by the device store): the
	   padding lifts the prompt clear of it, and the ResizeObserver above turns that into a refit. */
	.host {
		position: relative;
		flex: 1;
		min-height: 0;
		overflow: hidden;
		padding: var(--space-2);
		padding-bottom: calc(var(--space-2) + var(--kb-inset, 0px));
		/* Touch scroll belongs to xterm's own viewport; nothing here claims the gesture. */
		touch-action: pan-y;
	}
	.choices {
		display: flex;
		flex-wrap: wrap;
		gap: var(--space-3);
		justify-content: center;
		margin-top: var(--space-4);
	}
</style>
