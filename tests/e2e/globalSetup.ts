import type { FullConfig } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { BASE_PORT, BIN, E2E_HOME, LOG_DIR, REPO_ROOT } from './playwright.config';

const SHM = '/dev/shm';

type Backend = { child: ChildProcess; port: number; log: string };

/**
 * One PREBUILT backend per worker slot, on `BASE_PORT + slot`.
 *
 * This replaces `webServer`, which can only ever spawn one — and one backend is why the suite ran
 * `workers: 1`. A single backend is shared global state (`expectPristineWorkspace` and the save-path
 * guard in `lib/app.ts` are written against exactly that), so a second worker against it trips them:
 * measured, `--workers=4` on the shared backend failed 132 of 343. A backend per SLOT keeps the
 * contract intact rather than weakening it — a worker still owns its backend alone, so every
 * intra-worker ordering assumption survives; only a cross-worker one would not, and the suite has none.
 *
 * Two things move here because a fleet cannot do them per instance:
 *  · **the stale-`iox2` sweep**. It used to ride the `webServer` command; with N instances that is
 *    instance N deleting instance M's LIVE segments mid-run. It runs once, before the first spawn.
 *  · **backend stdout**. `webServer: { stdout: 'pipe' }` interleaved it into the report; N of those
 *    would be unreadable even if Playwright offered it. Each backend gets its own log file instead,
 *    and the directory is announced at start so a red test still has somewhere to look.
 */
export default async function spawnFleet(config: FullConfig): Promise<() => Promise<void>> {
	if (!fs.existsSync(BIN))
		throw new Error(`${BIN} is missing — \`npm run build:backend\` builds it and the SPA it serves`);
	// Stale segments only: nothing of ours is running yet, which is precisely why this belongs here
	// and cannot belong to an instance. A segment some other process owns is not ours to delete —
	// and not ours to collide with either, since iceoryx2 service names are pid-scoped.
	if (fs.existsSync(SHM))
		for (const f of fs.readdirSync(SHM).filter((n) => n.startsWith('iox2'))) {
			try {
				fs.rmSync(path.join(SHM, f));
			} catch {
				/* another process's segment */
			}
		}

	fs.mkdirSync(LOG_DIR, { recursive: true });
	// A test-scoped `GOOFI_HOME`, wiped up front: the fleet's session files and the test agent
	// config land here, never in the runner's real home. The fleet dies by SIGKILL, so the files
	// it leaves are exactly what a reader's probe must sweep — deliberately not cleaned up here.
	const home = E2E_HOME;
	fs.rmSync(home, { recursive: true, force: true });
	fs.mkdirSync(path.join(home, '.goofi'), { recursive: true });
	// `_sh` is a CONFIG entry a test writes, exactly as a user's own entry would be; the servers'
	// own seeding is absent-only, so this file stands.
	fs.writeFileSync(
		path.join(home, '.goofi', 'config.toml'),
		'[[agents]]\nname = "_sh"\ncommand = "sh"\n'
	);
	const fleet: Backend[] = [];
	for (let slot = 0; slot < config.workers; slot++) {
		const port = BASE_PORT + slot;
		const log = path.join(LOG_DIR, `backend-${slot}.log`);
		const fd = fs.openSync(log, 'w');
		// From the repo root, so it serves `frontend/build/`. NOT detached: the fleet stays in the
		// runner's process group, so a Ctrl-C reaches it even before `reap` gets its turn.
		// `--debug` opens `/dev/*`, which the integrity sweep drives: the gallery is a real route of
		// the shipped app, reachable only when it is asked for.
		const child = spawn(BIN, ['--bind', '127.0.0.1', '--port', String(port), '--debug'], {
			cwd: REPO_ROOT,
			env: { ...process.env, GOOFI_HOME: home },
			stdio: ['ignore', fd, fd]
		});
		fs.closeSync(fd);
		fleet.push({ child, port, log });
	}
	// SIGKILL, not SIGTERM: nothing here has teardown to do that the next run's sweep does not
	// already cover, and a signal that cannot be ignored is what keeps a run from leaving a backend
	// on a fixed port for the next run to collide with.
	const reap = async () => {
		for (const { child } of fleet) child.kill('SIGKILL');
	};
	try {
		await Promise.all(fleet.map(serving));
		// The one e2e pin on the session records: a REAL binary spawn under a scoped GOOFI_HOME
		// writes `sessions/<id>.json` per server, each url naming the port it serves.
		const sessions = fs
			.readdirSync(path.join(home, '.goofi', 'sessions'))
			.map((f) => JSON.parse(fs.readFileSync(path.join(home, '.goofi', 'sessions', f), 'utf8')));
		for (const { port } of fleet)
			if (!sessions.some((s) => s.url === `http://127.0.0.1:${port}`))
				throw new Error(`no session file names :${port} — got ${JSON.stringify(sessions)}`);
	} catch (e) {
		await reap(); // a fleet that never came up is still a fleet to reap
		throw e;
	}
	console.log(`  ${fleet.length} backend(s) on :${BASE_PORT}-${BASE_PORT + fleet.length - 1} · logs in ${LOG_DIR}`);
	return reap;
}

/**
 * Resolve once this backend answers `/`, or throw carrying the tail of its own log. A backend that
 * never starts has no test to fail, so this is the one place where the stdout that left the report
 * would otherwise be missed entirely — the likeliest cause being a port already taken.
 */
async function serving({ child, port, log }: Backend): Promise<void> {
	const deadline = Date.now() + 60_000;
	let exited = false;
	child.once('exit', () => (exited = true));
	for (;;) {
		try {
			if ((await fetch(`http://127.0.0.1:${port}/`)).ok) return;
		} catch {
			/* not listening yet */
		}
		if (exited || Date.now() > deadline)
			throw new Error(
				`goofi on :${port} never served / — tail of ${log}:\n${fs.readFileSync(log, 'utf8').slice(-800)}`
			);
		await new Promise((r) => setTimeout(r, 25));
	}
}
