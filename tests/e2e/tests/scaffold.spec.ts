import { test, expect } from '@playwright/test';

// Temporary: proves the Playwright toolchain runs. Deleted once real specs exist (Task 5).
test('playwright toolchain runs', () => {
	expect(1 + 1).toBe(2);
});
