import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

const BRIDGE = process.env.GOOFI_BRIDGE ?? 'http://127.0.0.1:8000';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		port: 5173,
		strictPort: false,
		proxy: {
			'/control': { target: BRIDGE, ws: true, changeOrigin: true },
			'/data': { target: BRIDGE, ws: true, changeOrigin: true },
			'/api': { target: BRIDGE, changeOrigin: true }
		}
	}
});
