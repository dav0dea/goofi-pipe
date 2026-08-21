/** goofi's `LayoutHost` — the one place a layout gesture becomes a manager op plus its undo step. */
import type { LayoutHost, TabRef } from 'panelty';
import type { Direction, Workspace } from 'panelty';
import { captureNavContext } from '$lib/stores/navContext';
import { history } from './history.svelte';
import { getControl, type Control } from '$lib/api/control';
import { workspace } from 'panelty';
import type { OpName } from '$lib/api/ops';

/** How the host reaches the manager, and how it sees the strip it is naming a tab against. */
export interface HostDeps {
	control: () => Control;
	tabs: () => Workspace[];
}

export function goofiLayoutHost(deps: HostDeps): LayoutHost {
	/** Names an op is carrying right now: the replica lands only after the round trip, so gestures
	 * repeated faster than that would all claim the same free name. */
	const inFlight = new Set<string>();

	function claimName(): string {
		const taken = new Set([...deps.tabs().map((t) => t.name), ...inFlight]);
		let n = 1;
		while (taken.has(`Tab ${n}`)) n += 1;
		const name = `Tab ${n}`;
		inFlight.add(name);
		return name;
	}

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
		async addTab(opts): Promise<TabRef | null> {
			const name = claimName();
			// Grouped so a tab that arrives already showing its panel type is one ctrl-Z.
			try {
				return await history().transaction('Add tab', async () => {
					const born = await cmd<TabRef>('Add tab', 'add_tab', { name, index: opts?.index });
					if (born && opts?.panelType) {
						await cmd('Change panel', 'set_panel', { panel: born.panel, type: opts.panelType });
					}
					return born;
				});
			} finally {
				inFlight.delete(name);
			}
		},

		async removeTab(tab) {
			return landed(await cmd('Close tab', 'remove_tab', { tab }));
		},

		async renameTab(tab, name) {
			return landed(await cmd('Rename tab', 'rename_tab', { tab, name }));
		},

		async reorderTab(tab, toIndex) {
			return landed(await cmd('Reorder tabs', 'reorder_tab', { tab, to_index: toIndex }));
		},

		async splitPanel(panel, direction: Direction, placeBefore, ratio) {
			const fresh = await cmd<string>('Split panel', 'split_panel', {
				panel,
				direction,
				place_before: placeBefore,
				ratio
			});
			return typeof fresh === 'string' ? fresh : null;
		},

		async removePanel(panel) {
			return landed(await cmd('Close panel', 'remove_panel', { panel }));
		},

		async resizeSplit(split, fractions) {
			return landed(await cmd('Resize', 'resize_split', { split, fractions }));
		},

		async setPanel(panel, patch, label = 'Change panel') {
			return landed(await cmd(label, 'set_panel', { panel, ...patch }));
		},

		// Two ops, because a fresh tab has no split for a move to land in.
		async movePanel(subtree, to) {
			if ('newTab' in to) {
				const name = claimName();
				try {
					return landed(
						await cmd('Move panel to new tab', 'add_tab', { name, index: to.newTab, subtree })
					);
				} finally {
					inFlight.delete(name);
				}
			}
			return landed(
				await cmd('Move panel', 'insert_at_panel', {
					subtree,
					target: to.panel,
					direction: to.direction,
					place_before: to.placeBefore
				})
			);
		}
	};
}

/** The live host, wired to the socket and to the strip the replica currently draws. */
let _live: LayoutHost | null = null;
export function layoutHost(): LayoutHost {
	if (!_live) {
		_live = goofiLayoutHost({ control: getControl, tabs: () => workspace().state.workspaces });
	}
	return _live;
}
