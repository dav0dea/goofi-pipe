import { expect, type Page } from '@playwright/test';

/**
 * The agent-harness door for specs: spawn, read the roster, hand it back.
 *
 * Everything here goes through a RAW `/control` socket (the `ua-reset.spec.ts` idiom) because the
 * harness these specs use is `_sh` — Task 1's hidden test adapter, a plain `/bin/sh`, the same
 * `_`-prefixed idiom the node catalog uses. It has no door in the UI on purpose: the panel's
 * launcher lists DETECTED harnesses only. So nothing here needs an agent harness installed.
 */

/** One RPC over its own `/control` socket. */
export async function rawCall(page: Page, op: string, payload: unknown): Promise<any> {
	return page.evaluate(
		async ({ op, payload }) => {
			const ws = new WebSocket(`ws://${location.host}/control`);
			await new Promise((r) => (ws.onopen = r));
			const reply = await new Promise<any>((res) => {
				const id = Math.floor(Math.random() * 1e6);
				const on = (e: MessageEvent): void => {
					if (typeof e.data !== 'string') return;
					const m = JSON.parse(e.data);
					if (m.id !== id) return;
					ws.removeEventListener('message', on);
					res(m);
				};
				ws.addEventListener('message', on);
				ws.send(JSON.stringify({ id, op, payload }));
			});
			ws.close();
			return reply;
		},
		{ op, payload }
	);
}

export const spawnSh = async (page: Page): Promise<string> =>
	(await rawCall(page, 'agent start', { name: '_sh' })).result.instance_id;

/** An instance's state on the manager's roster, or 'gone' once it has been dismissed. */
export async function stateOf(page: Page, id: string): Promise<string> {
	const roster = (await rawCall(page, 'agent list', {})).result;
	return roster.instances.find((i: { id: string }) => i.id === id)?.state ?? 'gone';
}

/** Stop and then DISMISS an instance, leaving nothing on the roster for the next spec. The second
 * stop is what dismisses, and only an ALREADY-EXITED instance is dismissed by one — so it waits
 * for the exit first. */
export async function dismiss(page: Page, id: string): Promise<void> {
	await rawCall(page, 'agent stop', { instance: id });
	await expect.poll(() => stateOf(page, id), { timeout: 15_000 }).toMatch(/exited|gone/);
	if ((await stateOf(page, id)) === 'exited') await rawCall(page, 'agent stop', { instance: id });
	await expect.poll(() => stateOf(page, id), { timeout: 15_000 }).toBe('gone');
}
