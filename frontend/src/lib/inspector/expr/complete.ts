/**
 * goofi-aware completion for the param expression editor — registered as PYTHON LANGUAGE DATA, which
 * is the whole point of the sub-project.
 *
 *   pythonLanguage.data.of({ autocomplete: source })
 *
 * `autocompletion()` does not take one completion function: it collects every source registered as
 * language data at the cursor and merges the results into ONE ranked popup. So `nd()` / `globals.` /
 * `np.` completions are not bolted onto Python's — they ARE Python's, arriving through the library's
 * designed extension point, ranked beside `globalCompletion` and `localCompletionSource`. Nothing
 * here uses `override`, which is the switch that would replace those sources instead of joining them.
 *
 * Context comes from the LEZER TREE, never a regex, and that is a correctness decision rather than a
 * stylistic one: `max(nd('a` , `x + nd('` and `nd('a', '` are three different answers that a regex
 * over the raw text reads as one. `complete.test.ts` drives exactly those cases.
 *
 * The scope this completes is small and fully knowable — `crates/goofi-py/src/expr.rs` evaluates with
 * `{nd, t, np, globals}` and nothing else.
 */
import { syntaxTree } from '@codemirror/language';
import { pythonLanguage } from '@codemirror/lang-python';
import type { Completion, CompletionContext, CompletionSource } from '@codemirror/autocomplete';
import type { Extension } from '@codemirror/state';
import type { EditorState } from '@codemirror/state';
import type { SyntaxNode } from '@lezer/common';
import { CURATED_NUMPY } from './numpy';
import type { CatalogueNode, ExprCatalogue } from './catalogue';

/**
 * What a cursor position MEANS, and the range a completion there replaces. `null` (no context) is the
 * common case and the important one: it is what leaves the stock Python sources to answer alone.
 */
export type ExprContext =
	| {
			kind: 'node';
			from: number;
			to: number;
			/** The quote the literal opened with — reused to close it on accept. */
			quote: string;
			/** Whether the literal already has its closing quote. */
			terminated: boolean;
	  }
	| { kind: 'slot'; node: string; from: number; to: number }
	| { kind: 'globals'; from: number; to: number }
	| { kind: 'numpy'; from: number; to: number }
	| { kind: 'scope'; from: number; to: number };

const text = (state: EditorState, node: SyntaxNode): string =>
	state.doc.sliceString(node.from, node.to);

/** Is this node a call to `nd`? */
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

/** The string a `nd('…')` call names its node with, or null when it is not a plain literal we can
 *  read (a variable, an f-string, a concatenation — the completer says nothing rather than guessing). */
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
	// The literal's CONTENT — accepting replaces all of it, so a caret parked mid-name does not
	// leave the tail of the old one behind.
	const from = str.from + 1;
	const to = terminated ? str.to - 1 : str.to;
	return pos < from || pos > to ? null : { kind: 'node', from, to, quote, terminated };
}

/** The cursor is at `<something>.` or `<something>.parti…` — whose attribute do we know? */
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
		return null;
	}
	// `nd('x').` — the node name is right there in the tree, so the slots are THAT node's. One level
	// further out (`nd('x').out.`) is a slot's own attribute and none of our business.
	if (isNdCall(state, obj)) {
		const name = ndArgName(state, obj);
		return name === null ? null : { kind: 'slot', node: name, from, to };
	}
	return null;
}

/** Classify a cursor position. Pure over `(parsed document, offset)` — the unit-test seam (spec §3). */
export function exprContext(state: EditorState, pos: number): ExprContext | null {
	const node = syntaxTree(state).resolveInner(pos, -1);
	if (node.name === 'Comment') return null;
	if (node.name === 'String') return stringContext(state, node, pos);
	const member = memberContext(state, node, pos);
	if (member) return member;
	// A partially-typed bare name: the four injected symbols are the entire scope and nothing else in
	// the app tells the user they exist. Deliberately narrow to a real `VariableName` — after a dot the
	// name is a PropertyName (handled above) and must not see the scope.
	if (node.name === 'VariableName') return { kind: 'scope', from: node.from, to: node.to };
	return null;
}

/** The evaluator's injected scope (`expr.rs`'s `eval` globals), in the order it is worth learning. */
const SCOPE: Completion[] = [
	{ label: 'nd', type: 'function', detail: "node reference — nd('name')", boost: 1 },
	{ label: 't', type: 'variable', detail: 'seconds since start', boost: 1 },
	{ label: 'np', type: 'namespace', detail: 'numpy', boost: 1 },
	{ label: 'globals', type: 'namespace', detail: 'patch globals — globals.key', boost: 1 }
];

/**
 * One node-name entry. D-X10: `_NdProxy._bare()` raises *"is ambiguous: it has multiple outputs"* for
 * a multi-output node, and the completer already knows the arity — so it says so in the detail AND
 * steers there, closing the call and leaving the caret on the dot so the slot list opens next. A
 * single-output node needs none of that; it is usable bare.
 */
function nodeEntry(node: CatalogueNode, ctx: Extract<ExprContext, { kind: 'node' }>): Completion {
	const n = node.slots.length;
	const multi = n > 1;
	const detail = multi
		? `${n} outputs — bare use is ambiguous, pick a .slot`
		: n === 1
			? '1 output — usable bare'
			: 'no outputs';
	// Closing the quote (and the call) is only right while the literal is still open; when it is
	// already closed those characters are sitting past the caret.
	const apply = ctx.terminated
		? undefined
		: `${node.name}${ctx.quote}${multi ? ').' : ''}`;
	return { label: node.name, detail, type: 'variable', apply };
}

/** What the popup is offered at a classified position. Pure — the other half of the §3 seam. */
export function entriesFor(ctx: ExprContext, cat: ExprCatalogue): Completion[] {
	switch (ctx.kind) {
		case 'node':
			return cat.nodes.map((n) => nodeEntry(n, ctx));
		case 'slot': {
			const node = cat.nodes.find((n) => n.name === ctx.node);
			return (node?.slots ?? []).map((s) => ({
				label: s.name,
				detail: s.dtype,
				type: 'property'
			}));
		}
		case 'globals':
			return cat.globals.map((g) => ({ label: g.name, detail: g.type, type: 'variable' }));
		case 'numpy':
			return CURATED_NUMPY.map((e) => ({ label: e.name, type: e.type }));
		case 'scope':
			return SCOPE;
	}
}

/** The `CompletionSource` the two pure halves add up to. */
export function goofiCompletionSource(catalogue: () => ExprCatalogue): CompletionSource {
	return (ctx: CompletionContext) => {
		const where = exprContext(ctx.state, ctx.pos);
		if (!where) return null;
		const options = entriesFor(where, catalogue());
		return options.length ? { from: where.from, to: where.to, options } : null;
	};
}

/** THE seam (sketch §3): our source, registered as Python language data — so CodeMirror collects it
 *  alongside Python's own and ranks them into a single popup. */
export function goofiLanguageData(catalogue: () => ExprCatalogue): Extension {
	return pythonLanguage.data.of({ autocomplete: goofiCompletionSource(catalogue) });
}
