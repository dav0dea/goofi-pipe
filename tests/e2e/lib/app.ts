import type { Page } from '@playwright/test';

/** Resolve once the app is fully live: `window.goofi` is published (AppShell mounted) and the
 * node catalog has arrived over the control WS (`query.nodeTypes()` non-empty ⇒ `hello` landed).
 * This is the readiness gate every spec waits on before driving the façade. */
export async function waitForApp(page: Page): Promise<void> {
	await page.waitForFunction(
		() => {
			const g = (window as any).goofi;
			return !!g && (g.query.nodeTypes()?.length ?? 0) > 0;
		},
		undefined,
		{ timeout: 20_000 }
	);
}
