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

// The gallery specs, named rather than pattern-matched, because what makes one is the ROUTE it
// drives: `inspector.spec.ts` is a product spec and `touch-inspector.spec.ts` is a gallery one.
const GALLERY = /\/(ui-gallery|inspector-gallery)\.spec\.ts$/;
const GALLERY_TOUCH = /\/touch-(ui-gallery|inspector)\.spec\.ts$/;

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
	// Six projects share the top-level `use`. `default` runs every existing spec EXCEPT the
	// touch-scoped ones (fine-pointer desktop chrome); `touch` runs only `touch-*` under Pixel 7
	// emulation, whose hasTouch+isMobile+viewport flip (pointer:coarse)/(hover:none) true so the
	// coarse density floor engages.
	//
	// `touch-landscape` and `tablet` deliberately run a NAMED FEW of the touch specs rather than
	// the whole suite. Both orientations of phone and tablet are in scope (CLAUDE.md), but almost
	// everything the touch suite proves is driven by the coarse media query, which answers
	// identically at 412px and at 1080px: re-running the hit floors, the hover doors and the
	// long-press doors in three projects would triple the wall clock to re-measure a constant.
	// What genuinely differs is what FITS — the header's progressive overflow, the inspector's
	// clamp against its host, a point-anchored popover's clamp against the screen, and whether a
	// 360px-tall viewport leaves a canvas at all — and that is `touch-reflow.spec.ts`.
	//
	// `touch-modality.spec.ts` and `touch-placement.spec.ts` join it in `touch-landscape` for the
	// opposite reason: what they measure is a CONSTANT that must survive the orientation change.
	// The rule both are built on is that orientation picks only the anchor and INPUT MODALITY picks
	// the gesture and the affordance, so the same assertions have to come back the same answer in
	// both anchors — and running one file in the two projects is what makes a re-coupling fail by
	// name instead of going unnoticed. `touch-placement` is the gesture half of that rule: placing
	// a node by dragging its ghost is gated on `pointerType`, per event, and a media query anywhere
	// in that path would show up here as a landscape-only red. Neither is in `tablet`: that project
	// is portrait too, so it would re-measure `touch`'s answer rather than the other anchor.
	//
	// Tablet LANDSCAPE (1138×712) is not its own project: it is wider than the tablet portrait
	// geometry and narrower than `default`'s 1280, and every invariant in the reflow file is
	// monotone in width, so it can only land between two geometries already covered.
	//
	// Both new descriptors are Chromium ones (`iPad (gen 7)` would pull in WebKit, which nothing
	// else in this suite needs and which would have to be downloaded before the suite could run).
	//
	// The two GALLERY projects are the only `fullyParallel` ones, because they are the only ones
	// that can be: `/dev/ui` and `/dev/inspector` mount no AppShell, open no socket and name no
	// patch, so the isolation the product specs get from a `finally` is theirs by construction.
	// They are also ~90% fixed cost, so splitting them per TEST rather than per FILE is what lets
	// the fleet fill its slots with them (33.0s → 7.3s standalone, measured). Two projects and not
	// one because the touch pair proves the coarse doors and needs the Pixel 7 descriptor to do it;
	// `npm run gallery` runs both, which is the inner loop while working on `$lib/ui`.
	projects: [
		{ name: 'default', testIgnore: [/touch-.*\.spec\.ts/, GALLERY] },
		{
			name: 'touch',
			testMatch: /touch-.*\.spec\.ts/,
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
			testMatch: /touch-(reflow|modality|placement)\.spec\.ts/,
			use: { ...devices['Pixel 7 landscape'] }
		},
		{
			name: 'tablet',
			testMatch: /touch-reflow\.spec\.ts/,
			use: { ...devices['Galaxy Tab S4'] }
		}
	]
});
