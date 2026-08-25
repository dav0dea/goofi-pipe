import { describe, expect, it } from 'vitest';
import { nodeHealth } from './nodeHealth';

describe('nodeHealth', () => {
	it('reports ok for a healthy node', () => {
		expect(nodeHealth({}).kind).toBe('ok');
		expect(nodeHealth(null).kind).toBe('ok');
		expect(nodeHealth({ error: null }).kind).toBe('ok');
		expect(nodeHealth({}).title).toBe('running');
	});

	it('reports a code error with its message', () => {
		const h = nodeHealth({ error: 'boom at line 3' });
		expect(h.kind).toBe('error');
		expect(h.title).toBe('boom at line 3');
	});

	it('reports booting stages with a stage label, on the amber tone', () => {
		expect(nodeHealth({ stage: 'creating' })).toEqual({
			kind: 'booting',
			tone: 'warn',
			title: 'creating…',
			label: 'creating…',
			status: 'ini',
			hint: 'creating…',
			runtime: undefined,
			runtimeTitle: undefined
		});
		expect(nodeHealth({ stage: 'setup' })).toEqual({
			kind: 'booting',
			tone: 'warn',
			title: 'setting up…',
			label: 'setting up…',
			status: 'ini',
			hint: 'setting up…',
			runtime: undefined,
			runtimeTitle: undefined
		});
		expect(nodeHealth({ stage: 'ready' }).kind).toBe('ok');
	});

	it('abbreviates every state for the pill, and spells it out in the hint', () => {
		// Four kinds, four tokens — the pill is a few characters wide, so none of them spells a word.
		expect(nodeHealth({ stage: 'ready' }).status).toBe('run');
		expect(nodeHealth({ error: 'boom' }).status).toBe('err');
		expect(nodeHealth({ stage: 'setup' }).status).toBe('ini');
		expect(nodeHealth({ stage: 'error' }).status).toBe('off');

		// The hint is where the long form lives — for an error, that is the message itself, and
		// losing it to the runtime label would cost the only place the message is readable.
		expect(nodeHealth({ error: 'boom' }).hint).toBe('boom');
		expect(nodeHealth({ error: 'boom', runtime: 'subprocess' }).hint).toBe(
			'boom — Python, in a subprocess'
		);
		expect(nodeHealth({ stage: 'ready', runtime: 'native' }).hint).toBe('running — Rust, in-process');
	});

	it('carries a compact runtime token, on every health kind', () => {
		// The pill shows one token; the long form is the tooltip. A node that runs NOWHERE — a port,
		// a facade — reports no runtime, and must not invent one.
		expect(nodeHealth({ stage: 'ready', runtime: 'native' }).runtime).toBe('rs.ip');
		// Every token is `<language>.<where>`, both halves two letters — the tier is never hover-only.
		expect(nodeHealth({ stage: 'ready', runtime: 'in-process' }).runtime).toBe('py.ip');
		expect(nodeHealth({ stage: 'ready', runtime: 'subprocess' }).runtime).toBe('py.sp');
		expect(nodeHealth({ stage: 'ready', runtime: 'subprocess' }).runtimeTitle).toBe(
			'Python, in a subprocess'
		);
		expect(nodeHealth({ stage: 'ready' }).runtime).toBeUndefined();

		// A demoted node is usually ALSO erroring, so the token has to survive every branch.
		expect(nodeHealth({ stage: 'error', runtime: 'subprocess' }).runtime).toBe('py.sp');
		expect(nodeHealth({ error: 'boom', runtime: 'in-process' }).runtime).toBe('py.ip');
		expect(nodeHealth({ stage: 'setup', runtime: 'native' }).runtime).toBe('rs.ip');
	});

	it('a real error outranks a stuck booting stage (failed setup keeps its stage)', () => {
		expect(nodeHealth({ stage: 'setup', error: 'setup boom' }).kind).toBe('error');
	});

	it('a node with nothing running behind it reads as dead, not merely errored', () => {
		// `stage: 'error'` is the manager saying there is NO instance: the host failed to start, or
		// setup() raised and nothing runs against a node that never initialized. A node that ran and
		// raised keeps its stage, and the two get different indicators — one blinks.
		const dead = nodeHealth({ stage: 'error', error: 'ModuleNotFoundError: torch' });
		expect(dead.kind).toBe('dead');
		expect(dead.tone).toBe('error');
		expect(dead.title).toBe('ModuleNotFoundError: torch');
		// …and it still reads as dead when the manager had no message to go with it.
		expect(nodeHealth({ stage: 'error' }).kind).toBe('dead');
		expect(nodeHealth({ stage: 'ready', error: 'boom' }).kind).toBe('error');
	});
});
