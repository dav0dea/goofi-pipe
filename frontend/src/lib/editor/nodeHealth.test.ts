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
			label: 'creating…'
		});
		expect(nodeHealth({ stage: 'setup' })).toEqual({
			kind: 'booting',
			tone: 'warn',
			title: 'setting up…',
			label: 'setting up…'
		});
		expect(nodeHealth({ stage: 'ready' }).kind).toBe('ok');
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
