/**
 * goofi's implementation of the panel system's [`LayoutHost`] — the one place a layout gesture
 * becomes a manager command.
 *
 * The panel system raises intents and holds no tree; this turns each into one op over `/control`
 * and records the one undo step that goes with it. The manager captured the exact inverse, so the
 * client entry carries no payload — it marks that a step happened, with a label and the
 * `NavContext` to re-orient to. A refusal records nothing, so the two stacks stay 1:1.
 *
 * It also owns what a tab is CALLED. A name is a label, not addressing, and only something that can
 * see the whole strip can pick a free one — which is why `addTab` takes no name.
 */
import type { LayoutHost, TabRef } from 'tatami';
import type { Direction, Workspace } from 'tatami';
import { captureNavContext } from '$lib/stores/navContext';
import { history } from './history.svelte';
import { getControl, type Control } from '$lib/api/control';
import { workspace } from 'tatami';
import type { OpName } from '$lib/api/ops';

/** How the host reaches the manager, and how it sees the strip it is naming a tab against. */
export interface HostDeps {
	control: () => Control;
	tabs: () => Workspace[];
}

export function goofiLayoutHost(deps: HostDeps): LayoutHost {
	/** Names an op is CARRYING right now. A tab is addressed by id, but its label must still be
	 * free — and the replica does not update until the round trip lands, so a gesture repeated
	 * faster than that (six taps on ＋) would ask for the same free name six times and have five
	 * refused. Scoped to the flight rather than "until the name is seen": that is what the name is
	 * actually racing, and it leaves nothing behind for a patch load to strand. */
	const inFlight = new Set<string>();

	function claimName(): string {
		const taken = new Set([...deps.tabs().map((t) => t.name), ...inFlight]);
		// Numbered from 1, and the manager's own first tab is `Tab 1` too: a bare name followed by a
		// numbered series reads as two different kinds of thing in one strip.
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
		// --- tabs ------------------------------------------------------------
		async addTab(opts): Promise<TabRef | null> {
			const name = claimName();
			// Two ops when a type is given, grouped as ONE client entry whose two children pop the
			// two manager commands in order — so a tab that arrives showing X is one ctrl-Z.
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

		// --- panels ----------------------------------------------------------
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

		// The two landings are two ops, because a fresh tab has no split for a move to land in — which
		// is why `add_tab` adopts a subtree at all. One method still, so the panel system has one
		// gesture rather than a capability to probe for. `ratio` is left to the op's own half.
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

/** The live host, wired to the socket and to the strip the replica currently draws. Reading the
 * names back off the replica rather than holding a second copy is what keeps them one fact. */
let _live: LayoutHost | null = null;
export function layoutHost(): LayoutHost {
	if (!_live) {
		_live = goofiLayoutHost({ control: getControl, tabs: () => workspace().state.workspaces });
	}
	return _live;
}
