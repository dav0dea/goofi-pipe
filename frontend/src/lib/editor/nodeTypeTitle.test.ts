import { describe, it, expect } from 'vitest';
import { nodeTypeTitle } from './nodeTypeTitle';
import type { NodeTypeInfo } from '$lib/api/control';

function unavailable(reason: string): NodeTypeInfo {
	// Exactly the shape `schemas.rs` emits for a node file whose probe failed.
	return {
		type: 'Psd',
		category: 'unavailable',
		doc: `This node could not be loaded: ${reason}`,
		available: false,
		dynamic: true,
		missing_deps: [reason],
		input_slots: {},
		output_slots: {},
		params: {}
	};
}

describe('nodeTypeTitle', () => {
	it("uses the backend's doc for an available node", () => {
		const t: NodeTypeInfo = { ...unavailable('x'), available: true, doc: 'Rolling window.' };
		expect(nodeTypeTitle(t)).toBe('Rolling window.');
	});

	it('reports a missing dependency by name', () => {
		// `reason` IS a bare module name — but only for ModuleNotFoundError.
		expect(nodeTypeTitle(unavailable('scipy'))).toBe('This node could not be loaded: scipy');
	});

	it('does not call a syntax error a missing dependency', () => {
		// The probe's `reason` is the exception line for anything that is not an import
		// failure, so labelling every greyed entry "missing dependency" produces nonsense.
		// The backend already phrases both cases correctly in `doc`; use it.
		const t = unavailable("SyntaxError: expected ':'");
		expect(nodeTypeTitle(t)).not.toContain('missing dependency');
		expect(nodeTypeTitle(t)).toBe("This node could not be loaded: SyntaxError: expected ':'");
	});
});
