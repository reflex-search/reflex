import fs from 'node:fs/promises';

declare const __PULSE_ROOT__: string;

type IndexRow = { id: string; tab: string; hash: string };

/**
 * getStaticPaths() for one tab, from bundle/index.json written by `rfx pulse`.
 * `cacheKey` (the page JSON's hash) feeds Astro's incremental build: a page is
 * re-rendered only when its JSON changed. pulse.config.json, and therefore every
 * sidebar, is part of the module graph Astro hashes itself.
 */
export async function tabPaths(tab: string) {
	const rows: IndexRow[] = JSON.parse(await fs.readFile(`${__PULSE_ROOT__}/bundle/index.json`, 'utf8'));
	const prefix = `${tab}/`;
	return rows
		.filter((r) => r.tab === tab)
		.map((r) => ({
			params: { slug: r.id === tab ? undefined : r.id.slice(prefix.length) },
			props: { pageFile: `${__PULSE_ROOT__}/bundle/pages/${r.id}.json` },
			cacheKey: r.hash,
		}));
}
