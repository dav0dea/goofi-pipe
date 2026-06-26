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

	it('reports a crash distinctly, with exit code + restart count', () => {
		const h = nodeHealth({ crashed: true, crashExit: -9, restarts: 2 });
		expect(h.kind).toBe('crashed');
		expect(h.title).toBe('process crashed (exit -9) — restarting (×2)');
	});

	it('a crash outranks a stale code error (it is auto-recovering)', () => {
		expect(nodeHealth({ crashed: true, error: 'old error', restarts: 1 }).kind).toBe('crashed');
	});

	it('handles a crash with no exit code', () => {
		expect(nodeHealth({ crashed: true, restarts: 1 }).title).toBe('process crashed — restarting (×1)');
	});
});
