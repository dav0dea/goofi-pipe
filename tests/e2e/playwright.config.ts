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
	// Two projects share the top-level `use`. `default` runs every existing spec EXCEPT the
	// touch-scoped ones (fine-pointer desktop chrome); `touch` runs only `touch-*` under Pixel 7
	// emulation, whose hasTouch+isMobile+viewport flip (pointer:coarse)/(hover:none) true so the
	// coarse density floor engages. R extends the `touch` project.
	projects: [
		{ name: 'default', testIgnore: /touch-.*\.spec\.ts/ },
		{
			name: 'touch',
			testMatch: /touch-.*\.spec\.ts/,
			use: { ...devices['Pixel 7'] }
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
