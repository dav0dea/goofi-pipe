import type { Page } from '@playwright/test';

/**
 * One RPC over its OWN `/control` socket, independent of the app's client.
 *
 * This is the oracle the seam specs read backend truth with: the page's replica is what is under
 * test, so asking it what the backend holds would ask the accused to testify.
 */
export async function rawCall(page: Page, op: string, payload: unknown = {}): Promise<any> {
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

/** The manager's own document — the truth a replica is judged against. */
export async function backendDoc(page: Page): Promise<any> {
	return (await rawCall(page, 'session state')).result;
}

/** The uids the manager holds, sorted. */
export async function backendNodes(page: Page): Promise<string[]> {
	return Object.keys((await backendDoc(page)).nodes ?? {}).sort();
}

/** The uids the page's replica holds, sorted — the same question asked of the other half. */
export async function replicaNodes(page: Page): Promise<string[]> {
	return page.evaluate(() =>
		((window as any).goofi.query.graph().nodes as Array<{ uid: string }>).map((n) => n.uid).sort()
	);
}

/**
 * Arm a page so its control socket can be dropped and held down. Call BEFORE `goto`.
 *
 * A real close, not `setOffline`: Chromium leaves an established WebSocket open when the context
 * goes offline, so the client's `send` succeeds into a void, no close event fires, and the RPC it
 * registered is never settled. That fixture cannot express the failure it is here to catch.
 */
export async function armSocketControl(page: Page): Promise<void> {
	await page.addInitScript(() => {
		const w = window as any;
		const Native = window.WebSocket;
		w.__ctlSockets = [];
		w.__ctlBlocked = false;
		class Ctl extends Native {
			constructor(url: string | URL, protocols?: string | string[]) {
				const control = String(url).includes('/control');
				// While blocked, every reconnection attempt is pointed at a refused port, so the client
				// takes the same close-and-back-off path a dead server gives it.
				super(control && w.__ctlBlocked ? 'ws://127.0.0.1:1/control' : url, protocols as any);
				if (control) w.__ctlSockets.push(this);
			}
		}
		w.WebSocket = Ctl;
	});
}

/** Drop the control socket and refuse its reconnections until [`restoreSocket`]. */
export async function dropSocket(page: Page): Promise<void> {
	await page.evaluate(() => {
		const w = window as any;
		w.__ctlBlocked = true;
		for (const s of w.__ctlSockets) {
			try {
				s.close();
			} catch {
				/* already closed */
			}
		}
	});
}

/** Let the client's next retry reach the real server again. */
export async function restoreSocket(page: Page): Promise<void> {
	await page.evaluate(() => ((window as any).__ctlBlocked = false));
}

/** Whether the tab can currently reach the manager, asked the only way the app exposes: an RPC. */
export async function reachesManager(page: Page): Promise<boolean> {
	return page.evaluate(async () => {
		try {
			await (window as any).goofi.query.nodeTypes();
			await (window as any).goofi.commands.setNodePos('__no_such_node__', [0, 0]);
			return true;
		} catch (e: any) {
			// A refusal from the MANAGER means the socket carried it; only a socket error means down.
			return !/socket/i.test(String(e?.message ?? e));
		}
	});
}
