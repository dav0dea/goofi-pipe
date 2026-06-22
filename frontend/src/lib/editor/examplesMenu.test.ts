import { describe, expect, it } from 'vitest';
import type { FsEntry } from '$lib/api/control';
import { examplesToMenuItems } from './examplesMenu';

const ent = (name: string, kind: 'file' | 'dir' = 'file', is_gfi = true): FsEntry => ({
	name,
	path: '/ex/' + name,
	kind,
	is_gfi,
	hidden: false,
	size: 0,
	mtime: 0
});

describe('examplesToMenuItems', () => {
	it('maps .gfi entries to load actions, stripping the extension', () => {
		const loaded: string[] = [];
		const items = examplesToMenuItems(
			[ent('Oscillator.gfi'), ent('PSD_topomap.gfi')],
			(p) => loaded.push(p)
		);
		expect(items.map((i) => i.label)).toEqual(['Oscillator', 'PSD_topomap']);
		items[1].action!();
		expect(loaded).toEqual(['/ex/PSD_topomap.gfi']);
	});

	it('filters out directories and non-gfi files', () => {
		const items = examplesToMenuItems(
			[ent('sub', 'dir', false), ent('readme.txt', 'file', false), ent('a.gfi')],
			() => {}
		);
		expect(items.map((i) => i.label)).toEqual(['a']);
	});

	it('shows a disabled placeholder when there are no examples', () => {
		const items = examplesToMenuItems([], () => {});
		expect(items).toHaveLength(1);
		expect(items[0].disabled).toBe(true);
	});
});
