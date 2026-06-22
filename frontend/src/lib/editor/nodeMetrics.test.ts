import { describe, it, expect } from 'vitest';
import { NODE, BOUNDARY, nodeSurfaceSize } from './nodeMetrics';

describe('nodeSurfaceSize', () => {
	it('a node with one collapsed output slot is header + one unit tall, node-width wide', () => {
		const s = nodeSurfaceSize(0, [false]);
		expect(s.width).toBe(NODE.width);
		expect(s.height).toBe(NODE.header + NODE.unit);
	});

	it('an expanded output slot adds the viewer plot height', () => {
		const s = nodeSurfaceSize(0, [true]);
		expect(s.height).toBe(NODE.header + NODE.unit + NODE.viewer);
	});

	it('a slotless node is still at least header + one unit (the input connector pitch)', () => {
		const s = nodeSurfaceSize(0, []);
		expect(s.height).toBe(NODE.header + NODE.unit);
	});

	it('grows to fit many input connectors when inputs out-tall the output-slot stack', () => {
		const s = nodeSurfaceSize(5, []); // 5 input units, no outputs
		expect(s.height).toBe(NODE.header + 5 * NODE.unit);
	});

	it('uses the taller of the input body and the output-slot stack', () => {
		// 1 input (1 unit) vs 2 collapsed outputs (2 units) → outputs win.
		const s = nodeSurfaceSize(1, [false, false]);
		expect(s.height).toBe(NODE.header + 2 * NODE.unit);
	});

	it('exposes a boundary-pill size distinct from a node (for snap of In/Out pills)', () => {
		expect(BOUNDARY.width).toBeGreaterThan(0);
		expect(BOUNDARY.height).toBeGreaterThan(0);
		expect(BOUNDARY.height).toBeLessThan(NODE.header); // a pill is short, not node-sized
	});
});
