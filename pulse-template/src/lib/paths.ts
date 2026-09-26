import fs from 'node:fs/promises';
import { getCollection, getEntry } from 'astro:content';
import pulse from '../../pulse.config.json';

declare const __PULSE_ROOT__: string;
const pagesFromFs = (pulse as any).build?.pageSource === 'fs';

type IndexRow = { id: string; tab: string; hash: string };

/** Variant D: bundle/index.json (id, tab, content hash) written by Rust; bodies stay on disk. */
async function fsPaths(tab: string) {
	const rows: IndexRow[] = JSON.parse(await fs.readFile(`${__PULSE_ROOT__}/bundle/index.json`, 'utf8'));
	const prefix = `${tab}/`;
	return rows
		.filter((r) => r.tab === tab)
		.map((r) => ({
			params: { slug: r.id.slice(prefix.length) },
			props: { pageFile: `${__PULSE_ROOT__}/bundle/pages/${r.id}.json`, pageId: r.id },
			cacheKey: r.hash,
		}));
}

/**
 * getStaticPaths() for one tab. `cacheKey` feeds Astro's experimental
 * incremental build: a page is re-rendered only when its JSON or one of the
 * fragments it references changed (the module graph, which includes
 * pulse.config.json and therefore every sidebar, is hashed by Astro itself).
 */
export async function tabPaths(tab: string) {
	if (pagesFromFs) return fsPaths(tab);
	const pages = await getCollection('pages', (p) => p.data.tab === tab);
	const prefix = `${tab}/`;
	return Promise.all(
		pages.map(async (page) => {
			const parts = [page.digest ?? page.id];
			for (const b of page.data.blocks as any[]) {
				const ref = b.type === 'prose' ? b.fragment : b.type === 'symbol-card' ? b.doc : undefined;
				if (ref) parts.push((await getEntry('fragments', ref))?.digest ?? ref);
			}
			return { params: { slug: page.id.slice(prefix.length) }, props: { page }, cacheKey: parts.join(':') };
		}),
	);
}
