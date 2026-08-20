/**
 * The flat arrangement → render tree reconstruction. This is the replica half of the fifth CRDT
 * doc root: the manager holds every page, split and panel as one id-keyed entry naming its parent
 * and its order, and the tree the panel system renders is rebuilt from those pointers here.
 */
import { describe, expect, it } from 'vitest';
import {
	buildWorkspaces,
	childIds,
	firstPanelIn,
	tabOf,
	splitFractions,
	type Arrangement
} from './arrangement';
import { arrangementEntries } from '$lib/crdt/graphDoc';

/** The manager's default arrangement, as `Layout::to_json` writes it. */
function defaultArr(): Arrangement {
	return {
		'page-1': { kind: 'tab', order: 0, name: 'Tab 1' },
		'panel-2': { kind: 'panel', order: 0, parent: 'page-1', size: 1, panel_type: 'node-editor' }
	};
}

/** One page holding a row split of two panels, the second bound to a node. */
function splitArr(): Arrangement {
	return {
		'page-1': { kind: 'tab', order: 0, name: 'Tab 1' },
		'split-4': { kind: 'split', order: 0, parent: 'page-1', size: 1, axis: 'row' },
		'panel-2': { kind: 'panel', order: 0, parent: 'split-4', size: 0.75, panel_type: 'node-editor' },
		'panel-3': {
			kind: 'panel',
			order: 1,
			parent: 'split-4',
			size: 0.25,
			panel_type: 'viewer',
			state: '{"node":"a1b2","slot":"out"}'
		}
	};
}

describe('buildWorkspaces', () => {
	it('turns a page holding one panel into one workspace', () => {
		const ws = buildWorkspaces(defaultArr());
		expect(ws).toHaveLength(1);
		expect(ws[0]).toMatchObject({ id: 'page-1', name: 'Tab 1' });
		expect(ws[0].root).toEqual({ kind: 'panel', id: 'panel-2', panelType: 'node-editor', state: undefined });
	});

	it('rebuilds a split from parentId + orderIndex, in order, with its shares', () => {
		const root = buildWorkspaces(splitArr())[0].root;
		expect(root.kind).toBe('split');
		if (root.kind !== 'split') return;
		expect(root.direction).toBe('row');
		expect(root.children.map((c) => c.id)).toEqual(['panel-2', 'panel-3']);
		expect(root.sizes).toEqual([0.75, 0.25]);
	});

	it('orders children by orderIndex, not by key order', () => {
		const arr = splitArr();
		arr['panel-2'].order = 1;
		arr['panel-3'].order = 0;
		const root = buildWorkspaces(arr)[0].root;
		if (root.kind !== 'split') throw new Error('expected a split');
		expect(root.children.map((c) => c.id), 'order is the SSOT for sibling position').toEqual([
			'panel-3',
			'panel-2'
		]);
		expect(root.sizes, 'and the shares follow the children').toEqual([0.25, 0.75]);
	});

	it('parses a panel state bag out of its JSON string leaf', () => {
		const root = buildWorkspaces(splitArr())[0].root;
		if (root.kind !== 'split') throw new Error('expected a split');
		expect(root.children[1]).toMatchObject({ state: { node: 'a1b2', slot: 'out' } });
	});

	it('orders pages by orderIndex into the tab strip', () => {
		const arr: Arrangement = {
			...defaultArr(),
			'page-7': { kind: 'tab', order: 0, name: 'Second' },
			'panel-8': { kind: 'panel', order: 0, parent: 'page-7', size: 1, panel_type: 'console' }
		};
		arr['page-1'].order = 1;
		expect(buildWorkspaces(arr).map((w) => w.name)).toEqual(['Second', 'Tab 1']);
	});

	it('hands back a default page while the replica is still empty', () => {
		const ws = buildWorkspaces({});
		expect(ws, 'an unsynced replica still renders something').toHaveLength(1);
		expect(ws[0].root.kind).toBe('panel');
	});

	it('drops a page whose root went missing rather than rendering a hole', () => {
		const arr: Arrangement = {
			...defaultArr(),
			'page-7': { kind: 'tab', order: 1, name: 'Second' }
		};
		expect(buildWorkspaces(arr).map((w) => w.id), 'no root ⇒ nothing to draw').toEqual(['page-1']);
	});
});

describe('tabOf / childIds / splitFractions', () => {
	it('walks parents up to the page a panel lives on', () => {
		expect(tabOf(splitArr(), 'panel-3')).toBe('page-1');
		expect(tabOf(splitArr(), 'page-1')).toBe('page-1');
		expect(tabOf(splitArr(), 'nope')).toBeNull();
	});

	it('lists a parent’s children in order', () => {
		expect(childIds(splitArr(), 'split-4')).toEqual(['panel-2', 'panel-3']);
	});

	it('reads a split’s current shares in child order — the resize commit’s baseline', () => {
		expect(splitFractions(splitArr(), 'split-4')).toEqual([0.75, 0.25]);
	});

	it('finds the first panel inside a subtree — what a drop hands focus to', () => {
		// A drag names a subtree: a panel is one of itself, a dragged TAB names its page's root
		// split. Either way the panel the user is now working in is the first one inside it.
		expect(firstPanelIn(splitArr(), 'panel-3')).toBe('panel-3');
		expect(firstPanelIn(splitArr(), 'split-4')).toBe('panel-2');
		expect(firstPanelIn(splitArr(), 'nope')).toBeNull();
	});
});

describe('arrangementEntries', () => {
	it('reads the root’s ENTRIES, skipping the id counter riding beside them', () => {
		const doc = { arrangement: { ...defaultArr(), '#seq': 2 } };
		const entries = arrangementEntries(doc);
		expect(Object.keys(entries).sort(), '`#seq` is a number, not an entry').toEqual([
			'page-1',
			'panel-2'
		]);
		expect(entries['panel-2']).toMatchObject({ kind: 'panel', parent: 'page-1', size: 1 });
	});

	it('round-trips a mirrored arrangement into the same tree the manager drew', () => {
		const root = buildWorkspaces(arrangementEntries({ arrangement: splitArr() }))[0].root;
		if (root.kind !== 'split') throw new Error('expected a split');
		expect(root.children.map((c) => c.id)).toEqual(['panel-2', 'panel-3']);
	});
});
