/** The one derivation of a node's status dot, so every surface that draws one agrees. */
import type { StatusTone } from '$lib/ui';
import type { NodeRuntime } from '$lib/api/control';

/** The compact runtime token the status pill carries, and the long form for its tooltip. Every
 * token is `<language>.<where>`, both halves two letters, so the pill's width never moves. */
const RUNTIME: Record<NodeRuntime, { token: string; title: string }> = {
	native: { token: 'rs.ip', title: 'Rust, in-process' },
	'in-process': { token: 'py.ip', title: 'Python, in-process' },
	subprocess: { token: 'py.sp', title: 'Python, in a subprocess' }
};

/** The status token, one per kind — three characters each, so the pill's width does not move with
 * the state. The long form stays in the tooltip, which for an error is the message itself. */
const STATUS: Record<HealthKind, string> = {
	ok: 'run',
	error: 'err',
	booting: 'ini',
	dead: 'off'
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
	/** Compact status token for the pill — `kind` abbreviated, uppercased by the Badge. */
	status: string;
	/** The pill's tooltip: the status spelled out, plus the runtime when there is one. */
	hint: string;
	/** Stage label rendered next to the dot while `kind` is 'booting'. */
	label?: string;
	/** Compact runtime token for the status pill; absent for a node that runs nowhere. */
	runtime?: string;
	/** The runtime spelled out, for the pill's tooltip. */
	runtimeTitle?: string;
}

export function nodeHealth(node: HealthNode | null | undefined): Health {
	const rt = node?.runtime ? RUNTIME[node.runtime] : undefined;
	const base = { runtime: rt?.token, runtimeTitle: rt?.title };
	const done = (h: Omit<Health, 'status' | 'hint'>): Health => ({
		...h,
		status: STATUS[h.kind],
		hint: rt ? `${h.title} — ${rt.title}` : h.title
	});
	if (node?.stage === 'error')
		return done({ kind: 'dead', tone: 'error', title: node.error ?? 'not running', ...base });
	if (node?.error) return done({ kind: 'error', tone: 'error', title: node.error, ...base });
	if (node?.stage === 'creating' || node?.stage === 'setup') {
		const label = node.stage === 'creating' ? 'creating…' : 'setting up…';
		return done({ kind: 'booting', tone: 'warn', title: label, label, ...base });
	}
	return done({ kind: 'ok', tone: 'ok', title: 'running', ...base });
}
