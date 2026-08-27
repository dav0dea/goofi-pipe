import { defineConfig, devices } from '@playwright/test';
import os from 'node:os';
import path from 'node:path';

// ONE BACKEND PER WORKER, on consecutive ports from BASE_PORT. Playwright re-loads this config
// inside each worker and sets TEST_PARALLEL_INDEX there — the worker's SLOT, stable across a
// restart — so the port derives per worker without a spec knowing that a port exists.
// `globalSetup.ts` spawns the fleet against the same arithmetic and reaps it after.
export const BASE_PORT = Number(process.env.GOOFI_E2E_PORT ?? 8500);
const PORT = BASE_PORT + Number(process.env.TEST_PARALLEL_INDEX ?? 0);
export const BASE_URL = `http://127.0.0.1:${PORT}`;
export const REPO_ROOT = path.resolve(__dirname, '../..');
export const LOG_DIR = path.join(__dirname, 'test-results', 'backend');
export const BIN = path.join(REPO_ROOT, 'target/debug/goofi');
// The fleet's test-scoped home: session records and the test agent config land here.
export const E2E_HOME = path.join(__dirname, 'test-results', 'goofi-home');

// Half the cores, capped at 8, so a small machine scales DOWN rather than up; the other gate anyone
// running this also runs is `cargo test --workspace`, and a suite that starves a build is a suite
// people stop running.
const WORKERS = Number(
	process.env.GOOFI_E2E_WORKERS ?? Math.max(1, Math.min(8, Math.floor(os.cpus().length / 2)))
);

const INTEGRITY = /integrity\.spec\.ts$/;

export default defineConfig({
	testDir: './tests',
	// A worker owns its backend alone, so the specs that share one are the ones that landed on the
	// same worker, and they still run one after another.
	fullyParallel: false,
	workers: WORKERS,
	globalSetup: './globalSetup.ts',
	forbidOnly: !!process.env.CI,
	retries: 0,
	reporter: [['list'], ['html', { open: 'never' }]],
	// Every test here is a SESSION — one boot, then a long ordered walk — so the budget is a
	// session's, not an assertion's.
	timeout: 120_000,
	expect: { timeout: 10_000 },
	use: {
		baseURL: BASE_URL,
		headless: true,
		trace: 'on-first-retry'
	},
	// The suite is four situations, and the projects are the geometries they are asked in. Only the
	// integrity sweep is asked more than once: it is the responsive test, and re-asking it is what
	// makes it one.
	projects: [
		{ name: 'desktop', testIgnore: /touch\.spec\.ts$/ },
		{
			name: 'phone',
			testMatch: /(touch|integrity)\.spec\.ts$/,
			use: { ...devices['Pixel 7'] }
		},
		{
			name: 'phone-landscape',
			testMatch: INTEGRITY,
			use: { ...devices['Pixel 7 landscape'] }
		},
		{
			name: 'tablet',
			testMatch: INTEGRITY,
			use: { ...devices['Galaxy Tab S4'] }
		}
	]
});
