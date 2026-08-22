// The seam: the frontend's socket client and the backend's document, meeting in a real browser.
//
// Everything goofi can DO is reachable through the op vocabulary, and `goofi-tests` proves it there
// against the manager itself. What only a browser can answer is whether the two halves of the
// socket fit — without slack, so nothing the client issues fails to land, and without overlap, so
// nothing lands twice or is held in two places that can disagree.
//
// The oracle is a RAW `/control` socket of the test's own. Asking the page's replica what the
// backend holds would ask the accused to testify.

import { test, expect, type Browser, type Page } from '@playwright/test';
import { closeSplit, splitRight, waitForApp } from '../lib/app';
import {
	armSocketControl,
	backendDoc,
	backendNodes,
	dropSocket,
	reachesManager,
	replicaNodes,
	restoreSocket
} from '../lib/raw';
import { addNode, nodeParams, redo, undo, waitForNode } from '../lib/goofi';

/** The replica and the manager hold the same uids. Polled: the replica lags by a round trip. */
async function expectAgreement(page: Page, what: string): Promise<string[]> {
	let last: { replica: string[]; backend: string[] } = { replica: [], backend: [] };
	await expect
		.poll(
			async () => {
				last = { replica: await replicaNodes(page), backend: await backendNodes(page) };
				return JSON.stringify(last.replica) === JSON.stringify(last.backend);
			},
			{ message: `${what}: replica and manager hold the same nodes` }
		)
		.toBe(true);
	return last.backend;
}

/**
 * Remove every node the manager holds.
 *
 * Run at the START of a session as well as in its `finally`: a session that depends on the previous
 * spec's teardown having landed is a session that fails for another file's reasons. Twice in
 * thirteen full runs, one of these opened against a backend that was not yet empty.
 */
async function clearGraph(page: Page): Promise<void> {
	await page.evaluate(async () => {
		const g = (window as any).goofi;
		// Facades too: `graph().nodes` is LEAVES, so a sub-patch left behind would outlive a sweep
		// that only read that list — and take its members and ports with it.
		const uids = [
			...g.query.graph().nodes.map((n: { uid: string }) => n.uid),
			...Object.keys(g.query.instances())
		];
		if (uids.length) await g.commands.removeNodes(uids);
	});
	await expect.poll(async () => (await backendNodes(page)).length).toBe(0);
}

test.describe('the control socket', () => {
	test('every op a client issues lands exactly once, and the manager agrees after each', async ({
		page
	}) => {
		await page.goto('/');
		await waitForApp(page);
		await clearGraph(page);
		try {
			await test.step('a fresh tab is synced, and both halves hold nothing', async () => {
				expect(await page.evaluate(() => (window as any).goofi.query.docSynced())).toBe(true);
				expect(await expectAgreement(page, 'at rest')).toEqual([]);
			});

			let osc = '';
			await test.step('an added node reaches the manager, and reaches it once', async () => {
				osc = await addNode(page, 'Oscillator', 'inputs');
				await waitForNode(page, osc);
				const uids = await expectAgreement(page, 'after add_node');
				expect(uids, 'one op, one node — an echo applied twice would show here').toEqual([osc]);
			});

			let buf = '';
			await test.step('a second node and a link between them', async () => {
				buf = await addNode(page, 'Buffer', 'signal', [280, 0]);
				await waitForNode(page, buf);
				await page.evaluate(
					([a, b]) =>
						(window as any).goofi.commands.addLink({
							node_out: a,
							slot_out: 'out',
							node_in: b,
							slot_in: 'data'
						}),
					[osc, buf]
				);
				await expect
					.poll(async () => (await backendDoc(page)).links.length, {
						message: 'the link reached the manager'
					})
					.toBe(1);
				await expect
					.poll(() => page.evaluate(() => (window as any).goofi.query.graph().links.length))
					.toBe(1);
			});

			await test.step('a param edit round-trips to the manager’s own document', async () => {
				await page.evaluate(
					(u) => (window as any).goofi.commands.updateParam(u, 'oscillator', 'amplitude', 0.42),
					osc
				);
				await expect
					.poll(async () => (await backendDoc(page)).nodes[osc].params.oscillator.amplitude.value)
					.toBeCloseTo(0.42, 5);
				expect(
					(await nodeParams(page, osc))?.oscillator?.amplitude?.value,
					'and the replica reads what the manager holds, not what it sent'
				).toBeCloseTo(0.42, 5);
			});

			await test.step('an expression carries its flags across, not only its source', async () => {
				await page.evaluate(
					(u) =>
						(window as any).goofi.commands.setExpression(u, 'oscillator', 'frequency', '2 * 3', {
							enabled: true
						}),
					osc
				);
				await expect
					.poll(async () => (await backendDoc(page)).nodes[osc].params.oscillator.frequency.expr?.source)
					.toBe('2 * 3');
				const expr = (await backendDoc(page)).nodes[osc].params.oscillator.frequency.expr;
				expect(expr.enabled, 'the flag rode with the source').toBe(true);
			});

			await test.step('a global is patch state, and lands the same way', async () => {
				await page.evaluate(() => (window as any).goofi.commands.addGlobal('seam_probe', 7, 'float'));
				await expect
					.poll(async () => (await backendDoc(page)).globals.seam_probe?.value)
					.toBe(7);
			});

			await test.step('undo reaches the manager — the history is the manager’s, not the tab’s', async () => {
				const before = await backendNodes(page);
				await undo(page);
				await expect
					.poll(async () => (await backendDoc(page)).globals.seam_probe, {
						message: 'the undone global is gone from the manager'
					})
					.toBeUndefined();
				expect(await backendNodes(page), 'and nothing else moved').toEqual(before);
				await expectAgreement(page, 'after undo');
			});

			await test.step('…and redo puts it back, through the same door', async () => {
				await redo(page);
				await expect.poll(async () => (await backendDoc(page)).globals.seam_probe?.value).toBe(7);
				await expectAgreement(page, 'after redo');
			});

			await test.step('a sub-patch port is a node to every op, and to the manager', async () => {
				// The replica draws a port from its own uid, and the manager holds it in the same node
				// map as everything else — so a port has to answer to add_node, add_link, edit_node and
				// remove_node exactly as a leaf does. Nothing else in this suite crosses that seam.
				const scope = await page.evaluate(
					(us) => (window as any).goofi.commands.groupNodes(us, [0, 0]),
					[osc, buf]
				);
				const port = await page.evaluate(
					(s) => (window as any).goofi.commands.addBoundary(s, 'InArray', [0, 40]),
					scope
				);
				await expect
					.poll(async () => (await backendDoc(page)).nodes[port]?.type, {
						message: 'the port reached the manager as a node record'
					})
					.toBe('InArray');
				expect((await backendDoc(page)).nodes[port].scope, 'and it names its sub-patch').toBe(scope);

				// The replica draws the pill at that very uid, which is what makes the ops addressable.
				await expect
					.poll(() =>
						page.evaluate(
							([s, p]) => !!(window as any).goofi.query.instance(s)?.interface[p],
							[scope, port]
						)
					)
					.toBe(true);

				await page.evaluate(
					([p, b]) =>
						(window as any).goofi.commands.addLink({
							node_out: p,
							slot_out: 'value',
							node_in: b,
							slot_in: 'data'
						}),
					[port, buf]
				);
				await expect
					.poll(async () =>
						(await backendDoc(page)).links.some(
							(l: { node_out: string; node_in: string }) => l.node_out === port && l.node_in === buf
						)
					)
					.toBe(true);

				// Rename and move go through the ORDINARY node ops, not doors of their own.
				await page.evaluate((p) => (window as any).goofi.commands.renameNode(p, 'left'), port);
				await page.evaluate((p) => (window as any).goofi.commands.setNodePos(p, [7, 9]), port);
				await expect.poll(async () => (await backendDoc(page)).nodes[port].name).toBe('left');
				expect((await backendDoc(page)).nodes[port].pos).toEqual({ x: 7, y: 9 });

				// …and so does the delete, which takes the port's inner wire with it.
				await page.evaluate((p) => (window as any).goofi.commands.removeNodes([p]), port);
				await expect.poll(async () => (await backendDoc(page)).nodes[port]).toBeUndefined();
				await expect
					.poll(async () =>
						(await backendDoc(page)).links.some(
							(l: { node_out: string }) => l.node_out === port
						)
					)
					.toBe(false);

				await page.evaluate((s) => (window as any).goofi.commands.expandInstance(s), scope);
				await expect.poll(async () => (await backendDoc(page)).nodes[scope]).toBeUndefined();
			});

			await test.step('a removal leaves the two halves holding the same nothing', async () => {
				await page.evaluate(
					(us) => (window as any).goofi.commands.removeNodes(us),
					[osc, buf]
				);
				expect(await expectAgreement(page, 'after remove')).toEqual([]);
				await expect
					.poll(async () => (await backendDoc(page)).links.length, {
						message: 'and the link went with the nodes'
					})
					.toBe(0);
			});
		} finally {
			await clearGraph(page);
		}
	});

	test('two tabs on one manager converge, and each undoes only its own work', async ({
		page,
		browser
	}: {
		page: Page;
		browser: Browser;
	}) => {
		await page.goto('/');
		await waitForApp(page);
		await clearGraph(page);
		const second = await browser.newContext({ baseURL: page.url() });
		const other = await second.newPage();
		try {
			await other.goto('/');
			await waitForApp(other);

			let mine = '';
			await test.step('what this tab adds, the other tab sees', async () => {
				mine = await addNode(page, 'Oscillator', 'inputs');
				await expect
					.poll(() => replicaNodes(other), { message: 'the peer mirrored the add' })
					.toContain(mine);
			});

			let theirs = '';
			await test.step('…and the other way round', async () => {
				theirs = await addNode(other, 'Buffer', 'signal', [280, 0]);
				await expect.poll(() => replicaNodes(page)).toContain(theirs);
			});

			await test.step('both tabs and the manager hold ONE document', async () => {
				const truth = await backendNodes(page);
				expect(truth.sort()).toEqual([mine, theirs].sort());
				expect(await replicaNodes(page)).toEqual(truth);
				expect(await replicaNodes(other)).toEqual(truth);
			});

			await test.step('an undo here takes back MY node, and leaves the peer’s standing', async () => {
				// The manager filters undo by session, which is what makes two tabs independent editors
				// of one document rather than two hands on one stack.
				await undo(page);
				await expect
					.poll(() => backendNodes(page), { message: 'my add was rolled back' })
					.toEqual([theirs]);
				await expect
					.poll(() => replicaNodes(other), { message: 'and the peer saw the removal, not its own loss' })
					.toEqual([theirs]);
			});
		} finally {
			await clearGraph(page);
			await second.close();
		}
	});

	test('a tab that loses the socket rejoins on the manager’s document, not on its own', async ({
		page,
		browser
	}: {
		page: Page;
		browser: Browser;
	}) => {
		// The failure this exists for: a client that reconnects and MERGES its stale replica onto a
		// document that moved under it. Nothing about that is visible without a real drop, which is
		// why neither half's own suite can ask it — the backend never sees the tab go, and the
		// frontend's fake socket never really closed.
		await armSocketControl(page);
		await page.goto('/');
		await waitForApp(page);
		await clearGraph(page);
		const peerCtx = await browser.newContext({ baseURL: page.url() });
		const peer = await peerCtx.newPage();
		try {
			await peer.goto('/');
			await waitForApp(peer);

			let doomed = '';
			await test.step('the tab holds a node before it goes dark', async () => {
				doomed = await addNode(page, 'Oscillator', 'inputs');
				await waitForNode(page, doomed);
				await expectAgreement(page, 'before the drop');
			});

			await test.step('the socket drops for real, and the tab cannot reach the manager', async () => {
				await dropSocket(page);
				// OBSERVABLY down, or the mutations below race the drop and the recovery proves nothing.
				// The oracle is an RPC, because it is the only one the app has: `docSynced()` is a LATCH
				// — "this replica has pulled at least once" — and stays true through any disconnection.
				await expect
					.poll(() => reachesManager(page), {
						message: 'the tab lost the manager',
						timeout: 20_000
					})
					.toBe(false);
			});

			let survivor = '';
			await test.step('the document moves under it, through a tab that is still connected', async () => {
				survivor = await addNode(peer, 'Buffer', 'signal', [280, 0]);
				await peer.evaluate((u) => (window as any).goofi.commands.removeNode(u), doomed);
				await expect
					.poll(() => backendNodes(peer), { message: 'the manager holds only the peer’s node' })
					.toEqual([survivor]);
			});

			await test.step('…and the tab SAYS it is dark, rather than looking healthy', async () => {
				// The half of a disconnection the user experiences. `net-frame` is the only surface that
				// reports it; the agent façade exposes no connection state at all, so nothing else in
				// this file could have noticed the socket had gone.
				await expect(
					page.getByTestId('net-frame'),
					'a tab with no manager must not look like a working one'
				).toBeVisible();
			});

			await test.step('…while the dark tab still shows what it last knew', async () => {
				// The step that makes the next one mean something: the replica is genuinely STALE here,
				// holding a node the manager no longer has and missing one it does. Without this, a tab
				// that had silently agreed all along would pass the convergence assertion below.
				expect(
					await replicaNodes(page),
					'a disconnected replica keeps its last document — it does not blank'
				).toEqual([doomed]);
			});

			await test.step('and on rejoining it takes the manager’s document whole', async () => {
				await restoreSocket(page);
				await expect
					.poll(() => reachesManager(page), { message: 'the tab reconnected', timeout: 30_000 })
					.toBe(true);
				await expect
					.poll(() => replicaNodes(page), {
						message: 'the node it was holding is GONE — a merge onto a stale base would keep it',
						timeout: 30_000
					})
					.toEqual([survivor]);
				await expect(
					page.getByTestId('net-frame'),
					'and the alarm clears with the reconnection that ended it'
				).toHaveCount(0);
			});
		} finally {
			await restoreSocket(page).catch(() => {});
			await clearGraph(peer);
			await peerCtx.close();
		}
	});

	test('a client the manager cannot speak to is told, not left half-working', async ({ page }) => {
		// The one failure mode where BOTH halves' own suites stay green: `PROTOCOL_VERSION` is declared
		// by hand on each side, each comments that the other must be bumped with it, and a client one
		// version behind still connects. `contracts.rs` pins that the two numbers agree; this pins what
		// the client does when they do not.
		//
		// The socket is PROXIED to the real server, not faked — only the hello frame's version is
		// rewritten on the way through, so everything else is the manager's own traffic.
		await page.routeWebSocket(/\/control$/, (ws) => {
			const server = ws.connectToServer();
			server.onMessage((m) => {
				if (typeof m !== 'string') return ws.send(m);
				try {
					const framed = JSON.parse(m);
					const payload = framed?.payload ?? framed?.result;
					if (payload && typeof payload === 'object' && 'protocol_version' in payload) {
						payload.protocol_version = 999;
						return ws.send(JSON.stringify(framed));
					}
				} catch {
					/* not a JSON frame; pass it through untouched */
				}
				ws.send(m);
			});
			ws.onMessage((m) => server.send(m));
		});
		await page.goto('/');
		await expect(
			page.getByRole('alert'),
			'a tab that cannot speak the manager’s protocol says so instead of half-working'
		).toContainText(/out of date/i);
	});

	test('a data stream reaches the tab as bytes and decodes to the frame the node emitted', async ({
		page
	}) => {
		// The OTHER socket. `/data` is binary and its format is pinned by a golden in `goofi-tests`;
		// what only the browser can say is that the decoder and its worker are wired to it.
		await page.goto('/');
		await waitForApp(page);
		try {
			const osc = await addNode(page, 'Oscillator', 'inputs');
			await waitForNode(page, osc);
			await page.evaluate(
				(u) => (window as any).goofi.commands.updateParam(u, 'oscillator', 'sfreq', 64),
				osc
			);

			const read = () =>
				page.evaluate((u) => (window as any).goofi.query.frameSummary(u, 'out'), osc);
			await test.step('frames arrive on the slot the viewer subscribed', async () => {
				await expect
					.poll(read, { message: 'a decoded frame reached the tab', timeout: 30_000 })
					.not.toBeNull();
			});
			const summary = await read();

			await test.step('a SECOND viewer on the same slot is served, and closing it leaves the first', async () => {
				// "One data stream per (node, slot), whatever the viewer count" is the architecture's
				// load-bearing claim about viewers; `running.rs` proves the manager's half, folding six
				// subscribers to one and then to none. What only a browser can say is that a second
				// viewer does not disturb the first — a detach and re-attach inside one render tick is
				// exactly the batch that used to pass through zero and tear the stream down under every
				// other viewer of that slot.
				//
				// A SPLIT, not a second tab: switching tabs unmounts the first viewer, so the two would
				// never be live at once and the question would not be asked at all.
				const index = () =>
					page.evaluate((u) => (window as any).goofi.query.frameSummary(u, 'out'), osc);
				await expect.poll(index, { message: 'the canvas viewer is served' }).not.toBeNull();

				await splitRight(page);
				const fresh = await page.evaluate(() => {
					const g = (window as any).goofi;
					const panels = g.query.panels();
					const p = panels[panels.length - 1].panelId;
					g.commands.setPanelType(p, 'viewer');
					g.commands.bindNodeToPanel(p, g.query.graph().nodes[0].uid);
					return p;
				});
				expect(fresh, 'the split gave us a second panel to bind').toBeTruthy();
				await expect
					.poll(index, { message: 'both viewers live, and the slot still answers' })
					.not.toBeNull();

				await closeSplit(page);
				await page.evaluate(() => new Promise((r) => setTimeout(r, 400)));
				await expect
					.poll(
						async () => (await index())?.numeric?.max ?? null,
						{ message: 'the surviving viewer is STILL served after the other went', timeout: 20_000 }
					)
					.not.toBeNull();
			});

			await test.step('…and it decodes to a real signal, not to zeroes', async () => {
				expect(summary.dtype, 'the wire dtype survived — numpy spelling, little-endian f32').toBe('<f4');
				expect(summary.numeric, 'an array frame carries its numbers').toBeTruthy();
				for (const k of ['min', 'max', 'mean'] as const) {
					expect(Number.isFinite(summary.numeric[k]), `${k} is finite`).toBe(true);
				}
				expect(
					summary.numeric.max - summary.numeric.min,
					'a sine spans a range — an all-zero buffer is a decoder that ran and read nothing'
				).toBeGreaterThan(0);
			});
		} finally {
			await clearGraph(page);
		}
	});
});
