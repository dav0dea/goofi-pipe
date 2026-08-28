/** goofi-aware completion for the expression editor, registered as Python language data so CodeMirror
 *  ranks it into the same popup as its own sources. */
import { syntaxTree } from '@codemirror/language';
import { pythonLanguage } from '@codemirror/lang-python';
import type { Completion, CompletionContext, CompletionSource } from '@codemirror/autocomplete';
import type { Extension } from '@codemirror/state';
import type { EditorState } from '@codemirror/state';
import type { SyntaxNode } from '@lezer/common';
import { CURATED_NUMPY } from './numpy';
import type { CatalogueNode, ExprCatalogue } from './catalogue';

/** What a cursor position means, and the range a completion there replaces; `null` leaves the stock
 *  Python sources to answer alone. */
export type RefTarget = { self: true } | { name: string };

export type ExprContext =
	| {
			kind: 'node';
			from: number;
			to: number;
			/** The quote the literal opened with — reused to close it on accept. */
			quote: string;
			terminated: boolean;
			callClosed: boolean;
	  }
	| { kind: 'path'; target: RefTarget; from: number; to: number }
	| { kind: 'slot'; target: RefTarget; from: number; to: number }
	| { kind: 'group'; target: RefTarget; from: number; to: number }
	| { kind: 'param'; target: RefTarget; group: string; from: number; to: number }
	| { kind: 'globals'; from: number; to: number }
	| { kind: 'numpy'; from: number; to: number }
	| { kind: 'scope'; from: number; to: number };

const text = (state: EditorState, node: SyntaxNode): string =>
	state.doc.sliceString(node.from, node.to);

function isNdCall(state: EditorState, call: SyntaxNode | null): boolean {
	if (!call || call.name !== 'CallExpression') return false;
	const callee = call.firstChild;
	return !!callee && callee.name === 'VariableName' && text(state, callee) === 'nd';
}

/** An `ArgList`'s first ACTUAL argument, past the parens/commas and the parser's error markers. */
function firstArg(argList: SyntaxNode | null): SyntaxNode | null {
	for (let c = argList?.firstChild ?? null; c; c = c.nextSibling) {
		if (c.name !== '(' && c.name !== ')' && c.name !== ',' && c.name !== '⚠') return c;
	}
	return null;
}

/** The string a `nd('…')` call names its node with, or null when it is not a plain literal. */
function ndArgName(state: EditorState, call: SyntaxNode): string | null {
	const arg = firstArg(call.getChild('ArgList'));
	if (!arg || arg.name !== 'String') return null;
	const raw = text(state, arg);
	const q = raw[0];
	if (q !== "'" && q !== '"') return null;
	return raw.length > 1 && raw.endsWith(q) ? raw.slice(1, -1) : raw.slice(1);
}

/** The cursor is inside a string: a node name iff that string is `nd`'s FIRST argument. */
function stringContext(state: EditorState, str: SyntaxNode, pos: number): ExprContext | null {
	const argList = str.parent;
	if (!argList || argList.name !== 'ArgList') return null;
	if (!isNdCall(state, argList.parent)) return null;
	if (firstArg(argList)?.from !== str.from) return null;
	const raw = text(state, str);
	const quote = raw[0];
	if (quote !== "'" && quote !== '"') return null;
	const terminated = raw.length > 1 && raw.endsWith(quote);
	// The literal's CONTENT: accepting replaces all of it, tail of the old name included.
	const from = str.from + 1;
	const to = terminated ? str.to - 1 : str.to;
	if (pos < from || pos > to) return null;
	return { kind: 'node', from, to, quote, terminated, callClosed: !!argList.getChild(')') };
}

/** The reference a member chain hangs off — `nd('x')` or `me` — and the attributes between. */
function refHead(state: EditorState, obj: SyntaxNode): { target: RefTarget; path: string[] } | null {
	const path: string[] = [];
	let cur: SyntaxNode | null = obj;
	while (cur && cur.name === 'MemberExpression') {
		const prop = cur.getChild('PropertyName');
		if (!prop) return null;
		path.unshift(text(state, prop));
		cur = cur.firstChild;
	}
	if (!cur) return null;
	if (cur.name === 'VariableName' && text(state, cur) === 'me') return { target: { self: true }, path };
	if (isNdCall(state, cur)) {
		const name = ndArgName(state, cur);
		return name === null ? null : { target: { name }, path };
	}
	return null;
}

/** The cursor is at `<something>.` — whose attribute do we know? */
function memberContext(state: EditorState, node: SyntaxNode, pos: number): ExprContext | null {
	let from = pos;
	let to = pos;
	if (node.name === 'PropertyName') {
		from = node.from;
		to = node.to;
	} else if (node.name !== '.') return null;
	const member = node.parent;
	if (!member || member.name !== 'MemberExpression') return null;
	const obj = member.firstChild;
	if (!obj) return null;
	if (obj.name === 'VariableName') {
		const name = text(state, obj);
		if (name === 'globals') return { kind: 'globals', from, to };
		if (name === 'np') return { kind: 'numpy', from, to };
	}
	const head = refHead(state, obj);
	if (!head) return null;
	const { target, path } = head;
	if (path.length === 0) return { kind: 'path', target, from, to };
	if (path[0] === 'out' && path.length === 1) return { kind: 'slot', target, from, to };
	if (path[0] === 'params' && path.length === 1) return { kind: 'group', target, from, to };
	if (path[0] === 'params' && path.length === 2)
		return { kind: 'param', target, group: path[1], from, to };
	// Deeper is the value's own attribute, and none of our business.
	return null;
}

/** Classify a cursor position. */
export function exprContext(state: EditorState, pos: number): ExprContext | null {
	const node = syntaxTree(state).resolveInner(pos, -1);
	if (node.name === 'Comment') return null;
	if (node.name === 'String') return stringContext(state, node, pos);
	const member = memberContext(state, node, pos);
	if (member) return member;
	// A real `VariableName` only — after a dot the name is a PropertyName and must not see the scope.
	if (node.name === 'VariableName') return { kind: 'scope', from: node.from, to: node.to };
	return null;
}

/** The evaluator's injected scope (`expr.rs`'s `eval` globals): goofi's own four names, then
 * `time()` and `from math import *` — the common slice of math's namespace, since the full list
 * lives Python-side and only eval is authoritative. */
const SCOPE: Completion[] = [
	{ label: 'nd', type: 'function', detail: "node reference — nd('name')", boost: 1 },
	{ label: 'me', type: 'variable', detail: 'this node — me.out / me.params', boost: 1 },
	{ label: 't', type: 'variable', detail: 'seconds since start', boost: 1 },
	{ label: 'np', type: 'namespace', detail: 'numpy', boost: 1 },
	{ label: 'globals', type: 'namespace', detail: 'patch globals — globals.key', boost: 1 },
	{ label: 'time', type: 'function', detail: 'wall-clock seconds — time()', boost: 1 },
	...'sin cos tan asin acos atan atan2 sinh cosh tanh exp expm1 log log2 log10 sqrt hypot floor ceil fabs fmod copysign degrees radians'
		.split(' ')
		.map((label) => ({ label, type: 'function', detail: 'math' })),
	...'pi e tau inf nan'.split(' ').map((label) => ({ label, type: 'constant', detail: 'math' }))
];

/** One node-name entry; a multi-output node steers to a `.slot`, since bare use of one raises. */
function nodeEntry(node: CatalogueNode, ctx: Extract<ExprContext, { kind: 'node' }>): Completion {
	const n = node.slots.length;
	const multi = n > 1;
	const detail = multi
		? `${n} outputs — bare use is ambiguous, pick .out.<slot>`
		: n === 1
			? '1 output — usable bare'
			: 'no outputs';
	// Finish only what is not written yet: an already-closed call's paren sits past the caret, so a
	// dot inserted here would land inside it.
	const tail = `${ctx.terminated ? '' : ctx.quote}${ctx.callClosed ? '' : `)${multi ? '.out.' : ''}`}`;
	return {
		label: node.name,
		detail,
		type: 'variable',
		apply: tail ? `${node.name}${tail}` : undefined
	};
}

/** The catalogue entry a reference target names — the editing node itself for `me`. */
function nodeOf(cat: ExprCatalogue, target: RefTarget): CatalogueNode | undefined {
	const name = 'self' in target ? cat.self : target.name;
	return name == null ? undefined : cat.nodes.find((n) => n.name === name);
}

/** What the popup is offered at a classified position. */
export function entriesFor(ctx: ExprContext, cat: ExprCatalogue): Completion[] {
	switch (ctx.kind) {
		case 'node':
			return cat.nodes.map((n) => nodeEntry(n, ctx));
		case 'path':
			return [
				{ label: 'out', type: 'namespace', detail: 'output slots — .out.<slot>' },
				{ label: 'params', type: 'namespace', detail: 'params — .params.<group>.<param>' }
			];
		case 'slot':
			return (nodeOf(cat, ctx.target)?.slots ?? []).map((s) => ({
				label: s.name,
				detail: s.dtype,
				type: 'property'
			}));
		case 'group':
			return (nodeOf(cat, ctx.target)?.params ?? []).map((g) => ({
				label: g.group,
				detail: `${g.names.length} param${g.names.length === 1 ? '' : 's'}`,
				type: 'namespace'
			}));
		case 'param':
			return (nodeOf(cat, ctx.target)?.params ?? [])
				.filter((g) => g.group === ctx.group)
				.flatMap((g) => g.names.map((n) => ({ label: n, type: 'property' })));
		case 'globals':
			return cat.globals.map((g) => ({ label: g.name, detail: g.type, type: 'variable' }));
		case 'numpy':
			return CURATED_NUMPY.map((e) => ({ label: e.name, type: e.type }));
		case 'scope':
			return SCOPE;
	}
}

function goofiCompletionSource(catalogue: () => ExprCatalogue): CompletionSource {
	return (ctx: CompletionContext) => {
		const where = exprContext(ctx.state, ctx.pos);
		if (!where) return null;
		const options = entriesFor(where, catalogue());
		return options.length ? { from: where.from, to: where.to, options } : null;
	};
}

export function goofiLanguageData(catalogue: () => ExprCatalogue): Extension {
	return pythonLanguage.data.of({ autocomplete: goofiCompletionSource(catalogue) });
}
