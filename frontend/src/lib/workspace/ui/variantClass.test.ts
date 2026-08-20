import { describe, it, expect } from 'vitest';
import {
	variantClass,
	type ButtonVariant,
	type ButtonSize,
	type ButtonDensity
} from './variantClass';

// variantClass is the one source of truth for the button primitives' variant/size
// class names (spec §3): a PURE mapper the components apply, kept unit-tested so the
// variant surface stays a closed union rather than being re-derived per component.
describe('variantClass', () => {
	const variants: ButtonVariant[] = ['default', 'primary', 'ghost', 'danger'];
	const sizes: ButtonSize[] = ['sm', 'md'];

	it('emits a namespaced class per variant', () => {
		expect(variantClass('default', 'md')).toContain('v-default');
		expect(variantClass('primary', 'md')).toContain('v-primary');
		expect(variantClass('ghost', 'md')).toContain('v-ghost');
		expect(variantClass('danger', 'md')).toContain('v-danger');
	});

	it('emits a namespaced class per size', () => {
		expect(variantClass('default', 'sm')).toContain('s-sm');
		expect(variantClass('default', 'md')).toContain('s-md');
	});

	it('returns exactly the variant + size classes, space-joined', () => {
		expect(variantClass('primary', 'sm')).toBe('v-primary s-sm');
		expect(variantClass('danger', 'md')).toBe('v-danger s-md');
	});

	it('never leaks another variant/size token into the string', () => {
		for (const variant of variants) {
			for (const size of sizes) {
				const cls = variantClass(variant, size);
				const others = variants.filter((v) => v !== variant);
				for (const other of others) expect(cls).not.toContain(`v-${other}`);
				expect(cls).toContain(`v-${variant}`);
				expect(cls).toContain(`s-${size}`);
			}
		}
	});

	it('is pure — identical inputs yield identical output', () => {
		expect(variantClass('ghost', 'md')).toBe(variantClass('ghost', 'md'));
	});

	// Density is a THIRD, orthogonal axis: `size` sets the glyph's type scale, density sets the
	// box floor. The comfortable default emits nothing extra (the primitives' base rule already
	// IS the comfortable box), so a Button's class list is untouched by its existence.
	describe('density', () => {
		const densities: ButtonDensity[] = ['comfortable', 'chrome'];

		it('defaults to comfortable and emits no density class for it', () => {
			expect(variantClass('ghost', 'sm')).toBe('v-ghost s-sm');
			expect(variantClass('ghost', 'sm', 'comfortable')).toBe('v-ghost s-sm');
		});

		it('emits a namespaced class for the chrome density', () => {
			expect(variantClass('ghost', 'sm', 'chrome')).toBe('v-ghost s-sm d-chrome');
		});

		it('is orthogonal — density never rewrites the variant or size class', () => {
			for (const variant of variants) {
				for (const size of sizes) {
					for (const density of densities) {
						const cls = variantClass(variant, size, density);
						expect(cls).toContain(`v-${variant}`);
						expect(cls).toContain(`s-${size}`);
					}
				}
			}
		});
	});
});
