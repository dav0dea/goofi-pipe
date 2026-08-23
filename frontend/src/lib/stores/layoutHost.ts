/** goofi's `LayoutHost` — the one place a layout gesture becomes a manager op plus its undo step. */
import type { LayoutHost, TabRef } from 'panelty';
import type { Direction } from 'panelty';
import { captureNavContext } from '$lib/stores/navContext';
import { history } from './history.svelte';
import { getControl, type Control } from '$lib/api/control';
import type { OpName } from '$lib/api/ops';

/** What `place_panel` answers: the entry it placed, and the tab it landed on. */
interface Placed {
	id: string;
	tab: string;
}

/** How the host reaches the manager. */
export interface HostDeps {
	control: () => Control;
}

/** Which SIDE of a target a drop lands on — the manager's spelling of the axis-and-half the panel
 * system raises as a pair. One word, so the two cannot disagree on the way over. */
const SIDE = { row: ['right', 'left'], column: ['bottom', 'top'] } as const;
const side = (d: Direction, placeBefore: boolean): string => SIDE[d][placeBefore ? 1 : 0];

export function goofiLayoutHost(deps: HostDeps): LayoutHost {
	/** Send one op and record ONE undo step for it. */
	async function cmd<T>(
		label: string,
		op: OpName,
		payload: Record<string, unknown>
	): Promise<T | null> {
		try {
			const res = await deps.control().call<T>(op, payload);
			if (!history().isSuspended) {
				history().record({ kind: 'graph_cmd', domain: 'graph', label, context: captureNavContext() });
			}
			return res;
		} catch (e) {
			console.warn(`${op} refused`, e);
			return null;
		}
	}

	const landed = (v: unknown): boolean => v !== null;

	return {
		// The NAME is the manager's: it sees the whole strip under the lock the add runs on, so
		// nothing here has to reserve one against a replica that lands a round trip later.
		async addTab(opts): Promise<TabRef | null> {
			// Grouped so a tab that arrives already showing its panel type is one ctrl-Z.
			return await history().transaction('Add tab', async () => {
				const born = await cmd<Placed>('Add tab', 'place_panel', { index: opts?.index });
				if (born && opts?.panelType) {
					await cmd('Change panel', 'edit_panel', { panel: born.id, type: opts.panelType });
				}
				return born && { tab: born.tab, panel: born.id };
			});
		},

		async removeTab(tab) {
			return landed(await cmd('Close tab', 'remove_panel', { panel: tab }));
		},

		async renameTab(tab, name) {
			return landed(await cmd('Rename tab', 'edit_panel', { panel: tab, name }));
		},

		async reorderTab(tab, toIndex) {
			return landed(await cmd('Reorder tabs', 'place_panel', { panel: tab, index: toIndex }));
		},

		// `to`, not `panel`: the fresh panel is what is placed, and it lands beside this one.
		async splitPanel(panel, direction: Direction, placeBefore, ratio) {
			const fresh = await cmd<Placed>('Split panel', 'place_panel', {
				to: panel,
				direction: side(direction, placeBefore),
				ratio
			});
			return fresh?.id ?? null;
		},

		async removePanel(panel) {
			return landed(await cmd('Close panel', 'remove_panel', { panel }));
		},

		async resizeSplit(split, fractions) {
			return landed(await cmd('Resize', 'edit_panel', { panel: split, fractions }));
		},

		async setPanel(panel, patch, label = 'Change panel') {
			return landed(await cmd(label, 'edit_panel', { panel, ...patch }));
		},

		// One op either way: a drop onto the tab bar names no target, a drop on an edge names one.
		async movePanel(subtree, to) {
			if ('newTab' in to) {
				return landed(
					await cmd('Move panel to new tab', 'place_panel', { panel: subtree, index: to.newTab })
				);
			}
			return landed(
				await cmd('Move panel', 'place_panel', {
					panel: subtree,
					to: to.panel,
					direction: side(to.direction, to.placeBefore)
				})
			);
		}
	};
}

/** The live host, wired to the socket. */
let _live: LayoutHost | null = null;
export function layoutHost(): LayoutHost {
	if (!_live) _live = goofiLayoutHost({ control: getControl });
	return _live;
}
