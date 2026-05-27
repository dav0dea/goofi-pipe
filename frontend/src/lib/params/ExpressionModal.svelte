<script lang="ts">
	import { onMount } from 'svelte';

	type Props = {
		title: string;
		initial: string;
		preview: string;
		onApply: (source: string) => void;
		onCancel: () => void;
	};
	const { title, initial, preview, onApply, onCancel }: Props = $props();

	let source = $state(initial);
	let textarea: HTMLTextAreaElement | null = $state(null);

	function apply(): void {
		onApply(source);
	}

	function onKeydown(e: KeyboardEvent): void {
		if (e.key === 'Escape') {
			e.preventDefault();
			e.stopPropagation();
			onCancel();
		} else if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
			e.preventDefault();
			e.stopPropagation();
			apply();
		}
	}

	onMount(() => {
		// Capture phase so Escape doesn't propagate to Editor's clear-selection
		// handler and so Cmd+Enter doesn't fire a TopBar shortcut.
		window.addEventListener('keydown', onKeydown, true);
		textarea?.focus();
		// Place caret at end so the user can keep typing where they left off.
		const len = source.length;
		textarea?.setSelectionRange(len, len);
		return () => window.removeEventListener('keydown', onKeydown, true);
	});
</script>

<div
	class="modal-overlay"
	role="presentation"
	onclick={onCancel}
	data-testid="expression-modal"
></div>
<div class="modal" role="dialog" aria-label="Edit expression: {title}">
	<header>
		<div class="title">edit expression: <span class="param">{title}</span></div>
		<button
			class="close"
			onclick={onCancel}
			aria-label="Close"
			data-testid="expression-modal-cancel"
		>
			✕
		</button>
	</header>
	<textarea
		bind:this={textarea}
		bind:value={source}
		spellcheck="false"
		autocapitalize="off"
		data-testid="expression-modal-textarea"
	></textarea>
	<footer>
		<div class="preview">
			<span class="hint">preview:</span>
			<span class="value">{preview}</span>
		</div>
		<div class="actions">
			<span class="kbd-hint">⌃⏎ apply · esc cancel</span>
			<button class="btn ghost" onclick={onCancel} data-testid="expression-modal-cancel-btn">
				cancel
			</button>
			<button class="btn primary" onclick={apply} data-testid="expression-modal-apply">
				apply
			</button>
		</div>
	</footer>
</div>

<style>
	.modal-overlay {
		position: fixed;
		inset: 0;
		background: color-mix(in srgb, var(--bg) 65%, transparent);
		backdrop-filter: blur(2px);
		z-index: 200;
	}
	.modal {
		position: fixed;
		left: 50%;
		top: 50%;
		transform: translate(-50%, -50%);
		width: min(640px, 90vw);
		max-width: 640px;
		max-height: 80vh;
		display: flex;
		flex-direction: column;
		background: var(--bg-elev-1);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		box-shadow: 0 12px 48px rgba(0, 0, 0, 0.6);
		z-index: 201;
	}
	header {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 10px 14px;
		border-bottom: 1px solid var(--border);
	}
	.title {
		flex: 1;
		font-family: var(--font-mono);
		font-size: 12px;
		color: var(--text-dim);
		letter-spacing: 0.02em;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.title .param {
		color: var(--accent);
	}
	.close {
		background: transparent;
		border: none;
		color: var(--text-faint);
		cursor: pointer;
		font-size: 14px;
		padding: 4px 8px;
		border-radius: 3px;
	}
	.close:hover {
		color: var(--text);
		background: var(--bg-elev-3);
	}
	textarea {
		flex: 1;
		min-height: 200px;
		resize: none;
		padding: 14px;
		font-family: var(--font-mono);
		font-size: 12px;
		line-height: 1.5;
		background: var(--bg);
		color: var(--accent);
		border: none;
		outline: none;
		tab-size: 4;
	}
	footer {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 8px 14px;
		border-top: 1px solid var(--border);
	}
	.preview {
		flex: 1;
		min-width: 0;
		display: flex;
		gap: 6px;
		align-items: baseline;
		font-family: var(--font-mono);
		font-size: 11px;
		color: var(--text-faint);
		overflow: hidden;
	}
	.preview .hint {
		opacity: 0.6;
	}
	.preview .value {
		color: var(--text-dim);
		font-variant-numeric: tabular-nums;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.actions {
		display: flex;
		gap: 8px;
		align-items: center;
	}
	.kbd-hint {
		font-family: var(--font-mono);
		font-size: 9px;
		color: var(--text-faint);
		opacity: 0.7;
		margin-right: 4px;
	}
	.btn {
		font-family: var(--font-mono);
		font-size: 11px;
		padding: 6px 12px;
		border-radius: 3px;
		cursor: pointer;
		letter-spacing: 0.02em;
		text-transform: lowercase;
		transition:
			background 80ms ease,
			color 80ms ease;
	}
	.btn.ghost {
		background: transparent;
		border: 1px solid var(--border);
		color: var(--text-dim);
	}
	.btn.ghost:hover {
		color: var(--text);
		border-color: var(--text-dim);
	}
	.btn.primary {
		background: var(--accent);
		border: 1px solid var(--accent);
		color: #0a0c10;
		font-weight: 600;
	}
	.btn.primary:hover {
		background: color-mix(in srgb, var(--accent) 80%, white);
	}
</style>
