import { describe, it, expect } from 'vitest';
import { NODE, BOUNDARY, nodeSurfaceSize, inputPorts, inputUnits } from './nodeMetrics';

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

describe('inputUnits (multi slots are 2× tall)', () => {
	const none = () => false;
	it('single slots count one unit each', () => {
		expect(inputUnits(['a', 'b', 'c'], none)).toBe(3);
	});
	it('a multi slot counts as two units', () => {
		expect(inputUnits(['a', 'b'], (s) => s === 'b')).toBe(3); // 1 + 2
	});
	it('floors at one so a slotless node still has a body', () => {
		expect(inputUnits([], none)).toBe(1);
	});
});

describe('inputPorts (stacked placement)', () => {
	const base = NODE.border + NODE.header;
	it('stacks single slots one unit apart, centred in each unit', () => {
		const ports = inputPorts(['a', 'b'], () => false);
		expect(ports.map((p) => p.units)).toEqual([1, 1]);
		expect(ports[0].top).toBe(base + NODE.unit / 2);
		expect(ports[1].top).toBe(base + NODE.unit + NODE.unit / 2);
	});
	it('a multi slot occupies two units and pushes the next slot down by two', () => {
		const ports = inputPorts(['m', 'n'], (s) => s === 'm');
		expect(ports[0].units).toBe(2);
		expect(ports[0].top).toBe(base + NODE.unit); // centre of a 2-unit block
		// 'n' starts after the 2-unit 'm' block.
		expect(ports[1].top).toBe(base + 2 * NODE.unit + NODE.unit / 2);
	});
	it('preserves slot order', () => {
		expect(inputPorts(['x', 'y', 'z'], () => false).map((p) => p.slot)).toEqual(['x', 'y', 'z']);
	});
});
