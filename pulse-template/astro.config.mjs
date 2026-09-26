// Pulse renderer template (M0a spike).
//
// Static-only: everything site-specific comes from ./pulse.config.json, which
// `rfx pulse` writes next to this file together with ./bundle/{pages,fragments}.
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig, passthroughImageService } from 'astro/config';
import { satteri } from '@astrojs/markdown-satteri';
import starlight from '@astrojs/starlight';
import { pulseMdastPlugin } from './src/plugins/remark-pulse.mjs';

const pulse = JSON.parse(fs.readFileSync(fileURLToPath(new URL('./pulse.config.json', import.meta.url)), 'utf8'));
const base = pulse.base ?? '/';
// Build knobs rfx can set per site (all optional).
const knobs = { incremental: true, chunkedStore: false, concurrency: 1, ...(pulse.build ?? {}) };

export default defineConfig({
	output: 'static',
	// Keep every cache next to the staged site (default is <root>/node_modules/.astro), so
	// rfx can persist/restore it as one directory and the site has no node_modules of its own.
	cacheDir: './.astro-cache',
	site: pulse.site,
	base,
	trailingSlash: 'always',
	image: { service: passthroughImageService() },
	// No dev toolbar, no telemetry prompts in CI: the build must never phone home.
	devToolbar: { enabled: false },
	vite: {
		cacheDir: './.astro-cache/vite',
		// Absolute site root for code that reads bundle/ at render time (build.pageSource = 'fs').
		define: { __PULSE_ROOT__: JSON.stringify(fileURLToPath(new URL('.', import.meta.url)).replace(/\/$/, '')) },
	},
	build: { format: 'directory', concurrency: knobs.concurrency },
	experimental: {
		// Re-render only pages whose getStaticPaths() cacheKey or module graph changed.
		incrementalBuild: knobs.incremental,
		// data-store.json passes ~150 MB at 5k pages; a single JSON string cannot exceed
		// V8's ~512 MB limit, so large sites must chunk the store.
		...(knobs.chunkedStore ? { collectionStorage: 'chunked' } : {}),
	},
	markdown: {
		processor: satteri({
			features: {
				// `# Errors {#sym-foo-errors}` lets Rust pick collision-free heading ids.
				headingAttributes: true,
				// Smart punctuation rewrites `--flag` to an en dash and <!-- --> to <!– —>:
				// doc comments are code-adjacent text and must render verbatim.
				smartPunctuation: false,
			},
			mdastPlugins: [pulseMdastPlugin({ base, sectionPrefixes: pulse.tabs.map((t) => t.prefix) })],
		}),
	},
	integrations: [
		starlight({
			title: pulse.title,
			description: pulse.description,
			pagefind: true,
			// Per-tab sidebars are injected by the route middleware; keep the global one empty so
			// Starlight does not build (and deep-clone) a sidebar per page for nothing.
			sidebar: [],
			tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 3 },
			routeMiddleware: './src/routeData.ts',
			components: {
				Header: './src/components/Header.astro',
			},
			customCss: [
				'@fontsource-variable/inter',
				'@fontsource/jetbrains-mono/400.css',
				'@fontsource/jetbrains-mono/700.css',
				'./src/styles/theme.css',
			],
			expressiveCode: {
				themes: ['github-dark', 'github-light'],
				useStarlightDarkModeSwitch: true,
				defaultProps: { wrap: true },
			},
			disable404Route: false,
		}),
	],
});
