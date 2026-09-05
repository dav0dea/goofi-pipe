import { describe, it, expect } from 'vitest';
import { paramGroupNames } from './ParamForm.svelte';
import type { NodeInstanceInfo } from '$lib/api/control';

function node(groups: string[]): NodeInstanceInfo {
	const params = Object.fromEntries(groups.map((g) => [g, {}]));
	return { params } as unknown as NodeInstanceInfo;
}

describe('param pages', () => {
	it('keep declared order, common last', () => {
		expect(paramGroupNames(node(['common', 'welch', 'psd', 'range']))).toEqual([
			'welch',
			'psd',
			'range',
			'common'
		]);
	});
});
