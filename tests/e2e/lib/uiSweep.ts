import fs from 'node:fs';
import path from 'node:path';
import type { Page, Locator } from '@playwright/test';

// The single source of truth for the primitive surface is `$lib/ui/index.ts`. The whole-library
// sweep (ui-gallery + touch-ui-gallery) is pinned to it so a future primitive added to the barrel
// without a gallery sample fails the sweep instead of silently escaping coverage.
const INDEX_TS = path.resolve(__dirname, '../../../frontend/src/lib/ui/index.ts');

/**
 * The primitive component names re-exported from `$lib/ui`, parsed straight from index.ts. A
 * primitive is a `export { default as X } from './X.svelte'` line; the named logic re-exports
 * (variantClass, layout, tabsState, …) and the `.svelte.ts` rune module carry no `default as`, so
 * they are excluded. Sorted for a stable set comparison against the sample registry.
 */
export function exportedPrimitives(): string[] {
	const src = fs.readFileSync(INDEX_TS, 'utf8');
	const names: string[] = [];
	const re = /export\s*\{([^}]*)\}\s*from\s*'\.\/[A-Za-z]+\.svelte'/g;
	for (let m = re.exec(src); m !== null; m = re.exec(src)) {
		const d = /default as (\w+)/.exec(m[1]);
		if (d) names.push(d[1]);
	}
	return names.sort();
}

/** A representative gallery sample for one primitive — its always-present testid plus, for the
 *  interactive ones, the actual tappable control the coarse `--hit` floor must land on. */
export interface Sample {
	/** A representative `/dev/ui` testid that is ALWAYS in the DOM (never behind an open/overlay state). */
	testid: string;
	/** Interactive controls take the keyboard-focus + coarse touch-target floors; static display
	 *  primitives (Badge, StatusDot, layout frames, …) do not. */
	interactive: boolean;
	/** The element the ≥44px coarse floor is measured on. Defaults to the testid element; overridden
	 *  where the real control is a descendant (a tab in the tablist, the summary in a Disclosure, the
	 *  range/select inside a Slider/Select wrapper). */
	control?: (page: Page) => Locator;
}

// Keyed by the exact `default as` export name — the sweep asserts these keys equal
// `exportedPrimitives()`, so this map cannot drift from the barrel.
export const SAMPLES: Record<string, Sample> = {
	// Interactive controls (focus + touch floors apply).
	Button: { testid: 'ui-button-default-md', interactive: true },
	IconButton: { testid: 'ui-icon-primary-md', interactive: true },
	NumberInput: { testid: 'ui-field-number', interactive: true },
	TextInput: { testid: 'ui-text-text', interactive: true },
	Trigger: { testid: 'ui-trigger', interactive: true },
	Toggle: { testid: 'ui-toggle', interactive: true },
	Chip: { testid: 'ui-chip', interactive: true },
	Popover: { testid: 'ui-popover-trigger', interactive: true },
	Dialog: { testid: 'ui-dialog-trigger', interactive: true },
	Slider: {
		testid: 'ui-slider',
		interactive: true,
		control: (page) => page.getByTestId('ui-slider').locator('input[type=range]')
	},
	Select: {
		testid: 'ui-select',
		interactive: true,
		control: (page) => page.getByTestId('ui-select').locator('select')
	},
	Tabs: {
		testid: 'ui-tabs',
		interactive: true,
		control: (page) => page.getByTestId('ui-tabs').getByRole('tab').first()
	},
	Disclosure: {
		testid: 'ui-disclosure',
		interactive: true,
		control: (page) => page.getByTestId('ui-disclosure').getByRole('button')
	},

	// Static display / layout primitives (rendered, never a tap target of their own).
	Stack: { testid: 'ui-stack-gap4', interactive: false },
	Row: { testid: 'ui-row-gap4', interactive: false },
	ScrollArea: { testid: 'ui-scrollarea', interactive: false },
	Bar: { testid: 'ui-bar', interactive: false },
	Field: { testid: 'ui-field-single', interactive: false },
	PanelShell: { testid: 'ui-panelshell', interactive: false },
	Badge: { testid: 'ui-badge-neutral', interactive: false },
	StatusDot: { testid: 'ui-statusdot-ok', interactive: false },
	Spinner: { testid: 'ui-spinner-md', interactive: false },
	EmptyState: { testid: 'ui-emptystate', interactive: false }
};

/** The tappable element the coarse `--hit` floor is measured on for an interactive sample. */
export function controlLocator(page: Page, sample: Sample): Locator {
	return sample.control ? sample.control(page) : page.getByTestId(sample.testid);
}
