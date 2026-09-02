/** The seed rule of the add-menu and its auto-link: the first slot of `t` on the seed's opposite
 *  side that the seed's slot may wire to, by the manager's one link rule. */
import type { NodeTypeInfo } from '$lib/api/control';
import { feeds, type SlotDtype } from '$lib/api/vocab';
import type { SlotClickSeed } from '$lib/stores/ui.svelte';

export function seedSlot(seed: SlotClickSeed, t: NodeTypeInfo): string | undefined {
	const candidates = seed.side === 'source' ? t.input_slots : t.output_slots;
	const may = (dt: string) =>
		seed.side === 'source'
			? feeds(seed.dtype as SlotDtype, dt as SlotDtype)
			: feeds(dt as SlotDtype, seed.dtype as SlotDtype);
	return Object.entries(candidates).find(([, dt]) => may(dt))?.[0];
}
