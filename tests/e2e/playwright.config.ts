import { defineConfig, devices } from '@playwright/test';
import path from 'node:path';

// Fixed test port (overridable) so webServer.url is concrete; one worker + one backend
// for determinism. The repo root is two levels up from this config.
const PORT = Number(process.env.GOOFI_E2E_PORT ?? 8399);
export const BASE_URL = `http://127.0.0.1:${PORT}`;
const REPO_ROOT = path.resolve(__dirname, '../..');

export default defineConfig({
	testDir: './tests',
	fullyParallel: false,
	workers: 1,
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
	// Four projects share the top-level `use`. `default` runs every existing spec EXCEPT the
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
	projects: [
		{ name: 'default', testIgnore: /touch-.*\.spec\.ts/ },
		{
			name: 'touch',
			testMatch: /touch-.*\.spec\.ts/,
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
	],
	// Spawn the PREBUILT binary from the repo root (so it serves frontend/build/ correctly),
	// clearing any stale iceoryx2 SHM first. `cargo build` happens via `npm run e2e` BEFORE
	// this, since webServer starts ahead of globalSetup. Playwright kills the process on teardown.
	webServer: {
		command: `bash -c "rm -f /dev/shm/iox2* 2>/dev/null || true; exec target/debug/goofi-pipe --bind 127.0.0.1 --port ${PORT}"`,
		cwd: REPO_ROOT,
		url: BASE_URL,
		reuseExistingServer: false,
		timeout: 60_000,
		stdout: 'pipe',
		stderr: 'pipe'
	}
});
