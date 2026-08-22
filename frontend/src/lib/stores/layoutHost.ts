/** goofi's `LayoutHost` — the one place a layout gesture becomes a manager op plus its undo step. */
import type { AddAt, Landing, LayoutHost } from 'panelty';
import { captureNavContext } from '$lib/stores/navContext';
import { history } from './history.svelte';
import { getControl, type Control } from '$lib/api/control';
import type { OpName } from '$lib/api/ops';

/** How the host reaches the manager. */
export interface HostDeps {
	control: () => Control;
}

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

	/** The placement half both `add_panel` and `move_panel` take, spelled once. */
	const where = (o: AddAt): Record<string, unknown> => ({
		direction: o.direction,
		place_before: o.placeBefore,
		ratio: o.ratio,
		index: o.index
	});

	return {
		async addPanel(at, opts = {}) {
			// Grouped so a panel that arrives already showing its type is one ctrl-Z.
			const born = await history().transaction(
				opts.direction ? 'Split panel' : 'Add tab',
				async () => {
					const id = await cmd<string>('Add panel', 'add_panel', { at, ...where(opts) });
					if (typeof id === 'string' && opts.panelType) {
						await cmd('Change panel', 'edit_panel', { panel: id, type: opts.panelType });
					}
					return id;
				}
			);
			return typeof born === 'string' ? born : null;
		},

		async removePanel(node) {
			return landed(await cmd('Close panel', 'remove_panel', { panel: node }));
		},

		async resizeSplit(split, fractions) {
			return landed(await cmd('Resize', 'edit_panel', { panel: split, fractions }));
		},

		async setPanel(panel, patch, label = 'Change panel') {
			return landed(await cmd(label, 'edit_panel', { panel, ...patch }));
		},

		async movePanel(subtree, to: Landing) {
			const at =
				'beside' in to
					? { to: to.beside, direction: to.direction, place_before: to.placeBefore }
					: { to: to.stack, index: to.index };
			return landed(await cmd('Move panel', 'move_panel', { panel: subtree, ...at }));
		}
	};
}

/** The live host, wired to the socket. */
let _live: LayoutHost | null = null;
export function layoutHost(): LayoutHost {
	if (!_live) _live = goofiLayoutHost({ control: getControl });
	return _live;
}
