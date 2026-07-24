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

	it('reports booting stages with a stage label', () => {
		expect(nodeHealth({ stage: 'creating' })).toEqual({
			kind: 'booting',
			title: 'creating…',
			label: 'creating…'
		});
		expect(nodeHealth({ stage: 'setup' })).toEqual({
			kind: 'booting',
			title: 'setting up…',
			label: 'setting up…'
		});
		expect(nodeHealth({ stage: 'ready' }).kind).toBe('ok');
	});

	it('a real error outranks a stuck booting stage (failed setup keeps its stage)', () => {
		expect(nodeHealth({ stage: 'setup', error: 'setup boom' }).kind).toBe('error');
	});

	it('a terminal boot error reads as an error', () => {
		// The supervisor sets stage 'error' + error text for a bootstrap failure.
		expect(nodeHealth({ stage: 'error', error: 'ModuleNotFoundError: torch' }).kind).toBe('error');
	});
});
