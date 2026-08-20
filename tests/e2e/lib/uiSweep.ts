import fs from 'node:fs';
import path from 'node:path';
import type { Page, Locator } from '@playwright/test';

// The single source of truth for the primitive surface is `$lib/ui/index.ts`. The whole-library
// sweep (ui-gallery + touch-ui-gallery) is pinned to it so a future primitive added to the barrel
// without a gallery sample fails the sweep instead of silently escaping coverage.
const INDEX_TS = path.resolve(__dirname, '../../../frontend/src/lib/ui/index.ts');

/**
 * The primitive component names re-exported from `$lib/ui`, parsed straight from index.ts.
 *
 * Two shapes, because the barrel now has two kinds of primitive. One the app owns and re-exports by
 * FILE (`export { default as X } from './X.svelte'`), and one the panel package owns and the barrel
 * takes back by NAME (`export { Button } from 'tatami'`) so the app still imports every primitive
 * from one place. A component is PascalCase in both, which is what separates it from the named
 * logic re-exports (isTextEditingTarget, MODE_ATTRS, ICONS, …) and from a `type` re-export. Sorted
 * for a stable set comparison against the sample registry.
 */
export function exportedPrimitives(): string[] {
	const src = fs.readFileSync(INDEX_TS, 'utf8');
	const names = new Set<string>();
	for (const [, clause, from] of src.matchAll(/export\s*\{([^}]*)\}\s*from\s*'([^']+)'/g)) {
		const byFile = /default as (\w+)/.exec(clause);
		if (byFile) {
			names.add(byFile[1]);
			continue;
		}
		// A re-export by name: every specifier that reads as a component, under its LOCAL name —
		// `TabStrip as Tabs` is `Tabs` to this app, and `Tabs` is what the gallery renders.
		if (from.endsWith('.svelte')) continue;
		for (const spec of clause.split(',')) {
			const t = spec.trim();
			if (!t || t.startsWith('type ')) continue;
			const local = (/(?:\bas\s+)?(\w+)$/.exec(t) ?? [])[1] ?? '';
			if (/^[A-Z][a-z]/.test(local)) names.add(local);
		}
	}
	return [...names].sort();
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
	ChoiceGrid: {
		testid: 'ui-choicegrid',
		interactive: true,
		control: (page) => page.getByTestId('ui-choicegrid').locator('button').first()
	},

	// Static display / layout primitives (rendered, never a tap target of their own).
	// Icon's sample is the whole vendored sheet — it draws every name in the table, so this one
	// testid also proves each icon's geometry actually renders.
	Icon: { testid: 'ui-icon-set', interactive: false },
	ScrollArea: { testid: 'ui-scrollarea', interactive: false },
	Bar: { testid: 'ui-bar', interactive: false },
	Field: { testid: 'ui-field-single', interactive: false },
	Badge: { testid: 'ui-badge-neutral', interactive: false },
	StatusDot: { testid: 'ui-statusdot-ok', interactive: false },
	EmptyState: { testid: 'ui-emptystate', interactive: false }
};

/** The tappable element the coarse `--hit` floor is measured on for an interactive sample. */
export function controlLocator(page: Page, sample: Sample): Locator {
	return sample.control ? sample.control(page) : page.getByTestId(sample.testid);
}
