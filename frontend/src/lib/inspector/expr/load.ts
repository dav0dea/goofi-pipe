/** The one edge CodeMirror enters the app across: keep it the only `@codemirror/*` import in `src/`. */
export type ExprEditorModule = typeof import('./editor');

let pending: Promise<ExprEditorModule> | null = null;

export function loadExprEditor(): Promise<ExprEditorModule> {
	return (pending ??= import('./editor'));
}
