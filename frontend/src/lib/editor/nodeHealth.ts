/** The one derivation of a node's status dot, so every surface that draws one agrees. */
import type { StatusTone } from '$lib/ui';
import type { NodeRuntime } from '$lib/api/control';

/** The compact runtime token the status pill carries, and the long form for its tooltip. */
const RUNTIME: Record<NodeRuntime, { token: string; title: string }> = {
	native: { token: 'rs', title: 'Rust, in-process' },
	'in-process': { token: 'py', title: 'Python, in-process' },
	subprocess: { token: 'py·sub', title: 'Python, in a subprocess' }
};

export type HealthKind = 'ok' | 'error' | 'booting' | 'dead';

export interface HealthNode {
	error?: string | null;
	stage?: 'creating' | 'setup' | 'ready' | 'error';
	runtime?: NodeRuntime;
}

export interface Health {
	kind: HealthKind;
	tone: StatusTone;
	title: string;
	/** Stage label rendered next to the dot while `kind` is 'booting'. */
	label?: string;
	/** Compact runtime token for the status pill; absent for a node that runs nowhere. */
	runtime?: string;
	/** The runtime spelled out, for the pill's tooltip. */
	runtimeTitle?: string;
}

export function nodeHealth(node: HealthNode | null | undefined): Health {
	const rt = node?.runtime ? RUNTIME[node.runtime] : undefined;
	const runtime = { runtime: rt?.token, runtimeTitle: rt?.title };
	if (node?.stage === 'error')
		return { kind: 'dead', tone: 'error', title: node.error ?? 'not running', ...runtime };
	if (node?.error) return { kind: 'error', tone: 'error', title: node.error, ...runtime };
	if (node?.stage === 'creating' || node?.stage === 'setup') {
		const label = node.stage === 'creating' ? 'creating…' : 'setting up…';
		return { kind: 'booting', tone: 'warn', title: label, label, ...runtime };
	}
	return { kind: 'ok', tone: 'ok', title: 'running', ...runtime };
}
