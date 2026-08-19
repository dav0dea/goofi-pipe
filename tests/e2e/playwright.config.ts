import { defineConfig, devices } from '@playwright/test';
import os from 'node:os';
import path from 'node:path';

// ONE BACKEND PER WORKER, on consecutive ports from BASE_PORT. Playwright re-loads this config
// inside each worker process and sets TEST_PARALLEL_INDEX there — the worker's SLOT, which is stable
// across a worker restart — so the port derives per worker without a single spec knowing that a port
// exists. `globalSetup.ts` spawns the fleet against the same arithmetic and reaps it after; slot 0
// is what the main process and a `--workers=1` debug run both read, so the base port stays the door
// in. 8500 rather than the old 8399 so a hand-run `cargo run` (:8000) cannot be what the fleet
// collides with, and the block 8500-8511 covers even a hand-raised GOOFI_E2E_WORKERS=12.
export const BASE_PORT = Number(process.env.GOOFI_E2E_PORT ?? 8500);
const PORT = BASE_PORT + Number(process.env.TEST_PARALLEL_INDEX ?? 0);
export const BASE_URL = `http://127.0.0.1:${PORT}`;
// The repo root is two levels up from this config. The fleet's logs go where Playwright's other
// artifacts go — it empties `test-results/` BEFORE globalSetup runs, so what is written there stays.
export const REPO_ROOT = path.resolve(__dirname, '../..');
export const LOG_DIR = path.join(__dirname, 'test-results', 'backend');

// Eight, not twelve. Measured on this 32-core machine, 12 workers finish in 51.1s against 55.6s at
// 8 — 8% for half again as many browsers and backends, against a tail that is one indivisible file
// either way. And the spare cores are not spare: `cargo test --workspace` is the other gate anyone
// running this suite is also running, and a suite that starves a concurrent build is a suite people
// stop running. Half the cores, capped at 8, so a small machine scales DOWN (12 backends + 12
// browsers is ~2.2GB) rather than up; GOOFI_E2E_WORKERS overrides it in either direction.
const WORKERS = Number(
	process.env.GOOFI_E2E_WORKERS ?? Math.max(1, Math.min(8, Math.floor(os.cpus().length / 2)))
);

// The gallery specs, matched on the exact file name because what makes one is the ROUTE it drives,
// not a word in its name: `inspector.spec.ts` is a product spec that happens to say "inspector".
const GALLERY = /\/gallery\.spec\.ts$/;
const GALLERY_TOUCH = /\/touch-gallery\.spec\.ts$/;

export default defineConfig({
	testDir: './tests',
	// Off at the top level, and that is load-bearing: a worker owns its backend alone, so the specs
	// that share one are the ones that landed on the same worker, and they still run one after
	// another. `expectPristineWorkspace` and every `finally` that hands the workspace back keep
	// working exactly as they did — what changed is only how many backends there are.
	fullyParallel: false,
	workers: WORKERS,
	globalSetup: './globalSetup.ts',
	forbidOnly: !!process.env.CI,
	retries: 0,
	reporter: [['list'], ['html', { open: 'never' }]],
	timeout: 30_000,
	expect: { timeout: 10_000 },
	use: {
		baseURL: BASE_URL,
		headless: true,
		trace: 'on-first-retry'
	},
	// Six projects share the top-level `use`. `default` is the fine-pointer desktop; `touch` runs
	// the `touch*` files under Pixel 7 emulation, whose hasTouch+isMobile+viewport flip
	// (pointer:coarse)/(hover:none) true so the coarse density floor engages.
	//
	// `touch-landscape` and `tablet` run a NAMED FEW of the touch files, not the whole set. Both
	// orientations of phone and tablet are in scope, but almost everything the touch suite proves is
	// driven by the coarse media query, which answers identically at 412px and at 1080px. What
	// genuinely differs is what FITS, and that is `touch-reflow.spec.ts`. `touch-anchor.spec.ts`
	// joins it in landscape for the opposite reason: what it measures is a CONSTANT that must
	// survive the orientation change — orientation picks only the anchor, input modality picks the
	// gesture — so running one file in two projects is what makes a re-coupling fail by name.
	// Neither is in `tablet`, which is portrait and would re-measure `touch`'s answer.
	//
	// Tablet LANDSCAPE (1138x712) is not its own project: it sits between tablet portrait and
	// `default`'s 1280, and every invariant in the reflow file is monotone in width.
	//
	// Both descriptors are Chromium ones (`iPad (gen 7)` would pull in WebKit, which nothing else
	// here needs and which would have to be downloaded before the suite could run).
	//
	// The two GALLERY projects are the only `fullyParallel` ones, because they are the only ones
	// that can be: `/dev/ui` and `/dev/inspector` mount no AppShell, open no socket and name no
	// patch, so the isolation the product specs get from a `finally` is theirs by construction.
	// They are also ~90% fixed cost, so splitting them per TEST rather than per FILE is what lets
	// the fleet fill its slots with them (33.0s -> 7.3s standalone, measured).
	projects: [
		{ name: 'default', testIgnore: [/touch.*\.spec\.ts/, GALLERY] },
		{
			name: 'touch',
			testMatch: /touch.*\.spec\.ts/,
			testIgnore: GALLERY_TOUCH,
			use: { ...devices['Pixel 7'] }
		},
		{ name: 'gallery', testMatch: GALLERY, fullyParallel: true },
		{
			name: 'gallery-touch',
			testMatch: GALLERY_TOUCH,
			fullyParallel: true,
			use: { ...devices['Pixel 7'] }
		},
		{
			name: 'touch-landscape',
			testMatch: /touch-(reflow|anchor)\.spec\.ts/,
			use: { ...devices['Pixel 7 landscape'] }
		},
		{
			name: 'tablet',
			testMatch: /touch-reflow\.spec\.ts/,
			use: { ...devices['Galaxy Tab S4'] }
		}
	]
});
