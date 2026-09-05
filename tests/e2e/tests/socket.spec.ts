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
	rawCall,
	backendNodes,
	dropSocket,
	reachesManager,
	replicaNodes,
	restoreSocket
} from '../lib/raw';
import { addNode, nodeParams, redo, selectNode, undo, waitForNode } from '../lib/goofi';

/** Whether the platform clipboard holds a goofi payload naming `uid` — the observable a copy is
 * finished by, since the manager is asked for the subtree before anything is written. */
async function clipboardHolds(page: Page, uid: string): Promise<boolean> {
	return page.evaluate(async (u) => {
		try {
			const text = await navigator.clipboard.readText();
			return text.includes('__goofi_clip__') && text.includes(u);
		} catch {
			return false;
		}
	}, uid);
}

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
		const uids = g.query.graph().nodes.map((n: { uid: string }) => n.uid);
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
				osc = await addNode(page, 'Oscillator');
				await waitForNode(page, osc);
				const uids = await expectAgreement(page, 'after add_node');
				expect(uids, 'one op, one node — an echo applied twice would show here').toEqual([osc]);
			});

			let buf = '';
			await test.step('a second node and a link between them', async () => {
				buf = await addNode(page, 'Buffer', [280, 0]);
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
				// Polled: the manager's delta reaches the replica by broadcast, after its own reply.
				await expect
					.poll(async () => (await nodeParams(page, osc))?.oscillator?.amplitude?.value, {
						message: 'and the replica reads what the manager holds, not what it sent'
					})
					.toBeCloseTo(0.42, 5);
			});

			await test.step('an expression carries its mode across, not only its text', async () => {
				await page.evaluate(
					(u) => (window as any).goofi.commands.setSource(u, 'oscillator', 'frequency', { expression: '2 * 3' }),
					osc
				);
				await expect
					.poll(async () => (await backendDoc(page)).nodes[osc].params.oscillator.frequency.expr)
					.toBe('2 * 3');
				const param = (await backendDoc(page)).nodes[osc].params.oscillator.frequency;
				expect(param.mode, 'the mode rode with the text').toBe('expression');
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

			await test.step('a delta landing mid-drag does not put the node back where it started', async () => {
				const card = page.locator(`.svelte-flow__node[data-id="${osc}"]`);
				const header = card.locator('.header');
				const start = (await header.boundingBox())!;
				const from = { x: start.x + start.width / 2, y: start.y + start.height / 2 };
				await page.mouse.move(from.x, from.y);
				await page.mouse.down();
				await page.mouse.move(from.x + 40, from.y + 30, { steps: 4 });
				await page.mouse.move(from.x + 120, from.y + 80, { steps: 8 });
				const held = (await card.boundingBox())!;
				expect(held.x, 'the drag moved it').toBeGreaterThan(start.x + 60);
				// ANOTHER writer's edit, with the pointer still down. The document holds where the drag
				// STARTED until the drop, so a replica that re-reads it over the gesture snaps the node back.
				await rawCall(page, 'node edit', { node: buf, name: 'renamedmiddrag' });
				await expect(page.locator(`.svelte-flow__node[data-id="${buf}"]`)).toContainText('renamedmiddrag');
				const after = (await card.boundingBox())!;
				expect(Math.abs(after.x - held.x), 'the dragged node stayed under the pointer').toBeLessThan(2);
				expect(Math.abs(after.y - held.y)).toBeLessThan(2);
				await page.mouse.up();
				await expect
					.poll(async () => (await backendDoc(page)).nodes[osc].pos.x, { timeout: 10_000 })
					.toBeGreaterThan(60);
				const rested = (await card.boundingBox())!;
				expect(Math.abs(rested.x - held.x), 'and the drop holds through the round trip').toBeLessThan(2);
			});

			await test.step('a sub-patch port is a node to every op, and to the manager', async () => {
				// The replica draws a port from its own uid, and the manager holds it in the same node
				// map as everything else — so a port has to answer to add_node, add_link, edit_node and
				// remove_node exactly as a leaf does. Nothing else in this suite crosses that seam.
				const scope = await page.evaluate(
					(us) => (window as any).goofi.commands.groupNodes(us, [0, 0]),
					[osc, buf]
				);
				// A port is born through `addNode` like anything else — `inst_id` is what makes it a
				// port OF that sub-patch rather than a node beside it.
				const port = await page.evaluate(
					(s) => (window as any).goofi.commands.addNode('InArray', [0, 40], s),
					scope
				);
				await expect
					.poll(async () => (await backendDoc(page)).nodes[port]?.type, {
						message: 'the port reached the manager as a node record'
					})
					.toBe('InArray');
				expect((await backendDoc(page)).nodes[port].scope, 'and it names its sub-patch').toBe(scope);

				// The replica draws the pill at that very uid, which is what makes the ops addressable —
				// and the facade wears it as a slot, because an in port FEEDS the sub-patch.
				await expect
					.poll(() =>
						page.evaluate(
							([s, p]) => !!(window as any).goofi.query.node(s)?.input_slots[p],
							[scope, port]
						)
					)
					.toBe(true);

				// …and the parent's facade grows the matching slot THERE AND THEN. A port IS the
				// sub-patch's slot; whether it is wired to a member yet is a separate question, and
				// gating the slot on that inner wire made an authored port invisible from outside.
				await expect(
					page.locator(`.svelte-flow__node[data-id="${scope}"] [data-handleid="${port}"]`),
					'the facade wears one handle per port, wired inside or not'
				).toHaveCount(1);

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

				// A cable from OUTSIDE names the port, and the canvas must reroute it onto the facade's
				// handle. The link and the drawing are two different questions, and only a browser
				// answers the second: the manager holds `feeder → port`, while the top level draws
				// `feeder → scope@port`, and nothing outside this file crosses that gap.
				const feeder = await addNode(page, 'Oscillator', [-200, 0]);
				await waitForNode(page, feeder);
				await page.evaluate(
					([f, sc, p]) =>
						(window as any).goofi.commands.addLink({
							node_out: f,
							slot_out: 'out',
							node_in: sc,
							slot_in: p
						}),
					[feeder, scope, port]
				);
				await expect
					.poll(async () =>
						(await backendDoc(page)).links.some(
							(l: { node_out: string; node_in: string }) => l.node_out === feeder && l.node_in === port
						),
						{ message: 'the manager stores the cable against the PORT' }
					)
					.toBe(true);
				await expect(
					page.locator(`.svelte-flow__edge[data-id="${feeder}.out\u2192${port}.value"]`),
					'and the canvas draws it, rerouted onto the facade'
				).toHaveCount(1);

				// …and it DRAWS as a node: entered, the port is a full node surface with a header and
				// a slot, not a chrome of its own. A pill had neither, and nothing else here would
				// notice a component going back to being one.
				await page.locator(`.svelte-flow__node[data-id="${scope}"]`).dblclick();
				await expect(page.getByTestId('subpatch-breadcrumb')).toBeVisible();
				const drawn = page.locator(`.svelte-flow__node[data-id="${port}"]`);
				await expect(drawn.locator('.header')).toBeVisible();
				await expect(
					drawn.locator('.svelte-flow__handle'),
					'an in port wears the one output slot its type declares'
				).toHaveCount(1);
				await page
					.getByTestId('subpatch-breadcrumb')
					.getByRole('button', { name: 'Patch', exact: true })
					.click();

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
				// The feeder was this step's alone; the step after this one counts what is left.
				await page.evaluate((f) => (window as any).goofi.commands.removeNodes([f]), feeder);
				await expect.poll(async () => (await backendDoc(page)).nodes[feeder]).toBeUndefined();
			});

			await test.step('a sub-patch is copied whole, and a selection is CUT into one', async () => {
				// The clipboard is the one seam neither half's suite can reach: `goofi-tests` proves
				// copy_nodes/paste_nodes against the manager, and the editor's own tests prove what
				// the store sends — but only a browser carries a payload OUT through the platform
				// clipboard and back in, which is what a copy and a paste actually are.
				await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
				const a = await addNode(page, 'Oscillator', [0, 300]);
				const b = await addNode(page, 'Buffer', [280, 300]);
				await waitForNode(page, a);
				await waitForNode(page, b);
				const scope = await page.evaluate(
					(us) => (window as any).goofi.commands.groupNodes(us, [140, 300]),
					[a, b]
				);
				await expect
					.poll(async () => (await backendDoc(page)).nodes[scope]?.type)
					.toBe('SubPatch');
				const inside = async (uid: string) =>
					Object.entries((await backendDoc(page)).nodes).filter(
						([, n]: [string, any]) => n.scope === uid
					).length;
				const held = await inside(scope);

				// Copy the FACADE and paste it. A sub-patch is not one type, so what rides the
				// clipboard has to be its members, its ports and the wiring among them.
				await selectNode(page, scope);
				await page.keyboard.press('Control+c');
				// Gate on the payload BEING there. A copy asks the manager for the subtree first,
				// so the clipboard is written a round trip after the key — and a paste that races
				// it reads whatever was there before.
				await expect
					.poll(() => clipboardHolds(page, scope), { message: 'the copy reached the clipboard' })
					.toBe(true);
				await page.keyboard.press('Control+v');
				await expect
					.poll(
						async () =>
							Object.values((await backendDoc(page)).nodes).filter(
								(n: any) => n.type === 'SubPatch'
							).length,
						{ message: 'the pasted sub-patch reached the manager' }
					)
					.toBe(2);
				const copy = Object.entries((await backendDoc(page)).nodes).find(
					([uid, n]: [string, any]) => n.type === 'SubPatch' && uid !== scope
				)![0];
				expect(await inside(copy), 'holding everything the original held').toBe(held);

				// CUT a plain selection, then paste it INSIDE a sub-patch. Cut is a copy and a
				// delete in one undo step; the paste lands where the editor is ENTERED, which is
				// the only way a node reaches the inside of a sub-patch by gesture.
				const c = await addNode(page, 'Oscillator', [0, 600]);
				const d = await addNode(page, 'Buffer', [280, 600]);
				await waitForNode(page, c);
				await waitForNode(page, d);
				await page.evaluate(
					([x, y]) =>
						(window as any).goofi.commands.addLink({
							node_out: x,
							slot_out: 'out',
							node_in: y,
							slot_in: 'data'
						}),
					[c, d]
				);
				// A two-node selection is SETUP here; the gesture under test is the cut and the paste.
				await page.evaluate((us) => (window as any).goofi.commands.select(us), [c, d]);
				await page.keyboard.press('Control+x');
				await expect
					.poll(() => clipboardHolds(page, c), { message: 'the cut reached the clipboard' })
					.toBe(true);
				await expect
					.poll(async () => (await backendDoc(page)).nodes[c], {
						message: 'a cut takes the nodes out of the patch'
					})
					.toBeUndefined();
				expect((await backendDoc(page)).nodes[d], 'both of them').toBeUndefined();

				await page.locator(`.svelte-flow__node[data-id="${scope}"]`).dblclick();
				await expect(page.getByTestId('subpatch-breadcrumb')).toBeVisible();
				await page.keyboard.press('Control+v');
				await expect
					.poll(async () => await inside(scope), {
						message: 'and the paste puts them INSIDE the sub-patch that was entered'
					})
					.toBe(held + 2);
				// …with the cable between them, because a link rides when both its ends do.
				const doc = await backendDoc(page);
				const within = Object.entries(doc.nodes)
					.filter(([, n]: [string, any]) => n.scope === scope)
					.map(([uid]) => uid);
				expect(
					doc.links.some(
						(l: { node_out: string; node_in: string; slot_in: string }) =>
							within.includes(l.node_out) && within.includes(l.node_in) && l.slot_in === 'data'
					),
					'the cut carried the wiring among the cut nodes'
				).toBe(true);
				await page
					.getByTestId('subpatch-breadcrumb')
					.getByRole('button', { name: 'Patch', exact: true })
					.click();

				// This step's scene was its own; the step after it counts what is left.
				await page.evaluate(
					(us) => (window as any).goofi.commands.removeNodes(us),
					[scope, copy]
				);
				await expect
					.poll(async () => (await backendDoc(page)).nodes[scope])
					.toBeUndefined();
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
				mine = await addNode(page, 'Oscillator');
				await expect
					.poll(() => replicaNodes(other), { message: 'the peer mirrored the add' })
					.toContain(mine);
			});

			let theirs = '';
			await test.step('…and the other way round', async () => {
				theirs = await addNode(other, 'Buffer', [280, 0]);
				await expect.poll(() => replicaNodes(page)).toContain(theirs);
			});

			await test.step('both tabs and the manager hold ONE document', async () => {
				const truth = await backendNodes(page);
				expect(truth.sort()).toEqual([mine, theirs].sort());
				await expect.poll(() => replicaNodes(page)).toEqual(truth);
				await expect.poll(() => replicaNodes(other)).toEqual(truth);
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
				doomed = await addNode(page, 'Oscillator');
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
				survivor = await addNode(peer, 'Buffer', [280, 0]);
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
			const osc = await addNode(page, 'Oscillator');
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
