/**
 * The one edge across which CodeMirror enters the app (D-X5).
 *
 * `./editor` is the root of the chunk: nothing else in `src/` imports a `@codemirror/*` module, so an
 * `await import()` here is what makes the editor a separate chunk rather than a tax on first paint of
 * a real-time signal app. `expression_enabled` is off by default on every param, so most sessions
 * never fetch it at all.
 *
 * ONE cached promise, module-scoped: the second fx field to render awaits the same fetch as the first,
 * and a field that mounts and unmounts does not re-request it.
 */
export type ExprEditorModule = typeof import('./editor');

let pending: Promise<ExprEditorModule> | null = null;

export function loadExprEditor(): Promise<ExprEditorModule> {
	return (pending ??= import('./editor'));
}
