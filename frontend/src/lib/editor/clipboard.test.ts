import { describe, it, expect } from 'vitest';
import { serializeClipboard, parseClipboard, clipToSpecs } from './clipboard';
import type { LinkInfo, NodeInstanceInfo } from '$lib/api/control';

function nodeInfo(uid: string, name: string, type = 'Oscillator'): NodeInstanceInfo {
	return {
		uid,
		name,
		type,
		category: 'inputs',
		doc: '',
		input_slots: { in: 'ARRAY' },
		output_slots: { out: 'ARRAY' },
		params: {},
		pos: [10, 20],
		viewers: {},
		membership: null,
		error: null
	};
}

describe('clipboard — node identity is the uid (links carry uid endpoints)', () => {
	it('keys instantiation specs by uid so paste can remap uid link endpoints', () => {
		// Display names deliberately differ from uids (the post-rekey reality).
		const nodes = [nodeInfo('uidA', 'oscillator0'), nodeInfo('uidB', 'buffer0', 'Buffer')];
		const links: LinkInfo[] = [
			{ node_out: 'uidA', slot_out: 'out', node_in: 'uidB', slot_in: 'in' }
		];
		const clip = serializeClipboard(nodes, links);
		const roundTripped = parseClipboard(JSON.stringify(clip));
		expect(roundTripped).not.toBeNull();

		const specs = clipToSpecs(roundTripped!, [100, 100]);
		// The spec key must be the uid — the same identity the link endpoints use —
		// so instantiateNodes' rename map (key -> newUid) resolves the endpoints.
		expect(specs.map((s) => s.key).sort()).toEqual(['uidA', 'uidB']);
	});
});
