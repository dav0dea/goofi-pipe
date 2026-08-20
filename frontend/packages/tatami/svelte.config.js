import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** The package ships SOURCE, so this config exists for `svelte-check` alone — it is what teaches it
 *  to read `lang="ts"` in a component. A consumer's own preprocessor compiles the files. */
export default { preprocess: vitePreprocess() };
