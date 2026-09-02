/** What the reference picker offers: the nodes with an output a param may reference, and those
 *  outputs — the typing rule the manager holds, applied before the pick. */
import type { ExprCatalogue } from './catalogue';

export interface PickerOption {
	label: string;
	detail?: string;
}

/** The output kind a param of `type` may reference: a string reads a STRING, everything else an ARRAY. */
export function wantedDtype(paramType: string): string {
	return paramType === 'string' ? 'STRING' : 'ARRAY';
}

export function refNodes(cat: ExprCatalogue, dtype: string): PickerOption[] {
	return cat.nodes
		.filter((n) => n.slots.some((s) => s.dtype === dtype))
		.map((n) => ({
			label: n.name,
			detail: n.slots.filter((s) => s.dtype === dtype).map((s) => s.name).join(', ')
		}));
}

export function refSlots(cat: ExprCatalogue, node: string, dtype: string): PickerOption[] {
	return (cat.nodes.find((n) => n.name === node)?.slots ?? [])
		.filter((s) => s.dtype === dtype)
		.map((s) => ({ label: s.name, detail: s.dtype }));
}

/** `node.slot` split at its one dot; a malformed or empty value is two empty halves. */
export function splitReference(reference: string | null): [string, string] {
	const at = reference?.indexOf('.') ?? -1;
	if (!reference || at < 0) return ['', ''];
	return [reference.slice(0, at), reference.slice(at + 1)];
}
