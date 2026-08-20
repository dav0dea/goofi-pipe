import { describe, it, expect } from 'vitest';
import { panelBinding } from './viewBinding';

describe('panelBinding', () => {
	it('resolves kind/settings from panel state and writes back via setState', () => {
		let state: Record<string, unknown> = { node: 'n', slot: 's' };
		const b = panelBinding(
			() => state,
			(s) => {
				// `set_panel` MERGES the state it is handed, so the double does too — that is
				// the seam, and a binding writes only the key it changed.
				state = { ...state, ...(s as Record<string, unknown>) };
			},
			'ARRAY'
		);
		expect(b.kind).toBe('line'); // default
		b.setKind('image');
		expect(state.kind).toBe('image');
		expect(state.node, 'a kind change names the kind and nothing else').toBe('n');
		expect(b.kind).toBe('image'); // re-resolves from updated state
		b.setSetting('colormap', 'viridis');
		expect((state.settings as Record<string, unknown>).colormap).toBe('viridis');
		expect(b.settings.colormap).toBe('viridis');
	});

	it('forces the string viewer for STRING dtype regardless of stored kind', () => {
		const state = { kind: 'image' };
		const b = panelBinding(
			() => state,
			() => {},
			'STRING'
		);
		expect(b.kind).toBe('string');
	});
});
