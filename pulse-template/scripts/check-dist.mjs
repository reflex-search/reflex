#!/usr/bin/env node
// Static correctness + size checks on a built Pulse site (M0a spike).
//
//   node scripts/check-dist.mjs <site_dir>          (reads <site_dir>/dist + pulse.config.json)
//
// Prints a JSON report; exits 1 when a correctness check fails.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

// Resolve ultrahtml (a Starlight dependency) by walking up from the site dir, the way
// Node resolves the site's own imports, so this runs against the extracted runtime.
function findUp(from, rel) {
	for (let d = path.resolve(from); ; d = path.dirname(d)) {
		const c = path.join(d, 'node_modules', rel);
		if (fs.existsSync(c)) return c;
		if (path.dirname(d) === d) throw new Error(`cannot find ${rel} above ${from}`);
	}
}
const { parse, walkSync, ELEMENT_NODE } = await import(pathToFileURL(findUp(process.argv[2] ?? '.', 'ultrahtml/dist/index.js')).href);

const site = path.resolve(process.argv[2] ?? '.');
const dist = path.join(site, 'dist');
const cfg = JSON.parse(fs.readFileSync(path.join(site, 'pulse.config.json'), 'utf8'));
const base = cfg.base.replace(/\/$/, '');

const html = [];
const all = [];
(function walk(d) {
	for (const e of fs.readdirSync(d, { withFileTypes: true })) {
		const p = path.join(d, e.name);
		if (e.isDirectory()) walk(p);
		else {
			all.push(p);
			if (e.name.endsWith('.html')) html.push(p);
		}
	}
})(dist);

const read = (p) => fs.readFileSync(p, 'utf8');
// --sizes-only (large N): per-file checks run on an even sample of ~300 pages.
const sizesOnly = process.argv.includes('--sizes-only');
const step = sizesOnly ? Math.max(1, Math.ceil(html.length / 300)) : 1;
const scan = html.filter((_, i) => i % step === 0);
const allScan = [...all.filter((f) => !f.endsWith('.html')), ...scan];
const rel = (p) => path.relative(dist, p);
const size = (p) => fs.statSync(p).size;
const results = [];
const check = (name, ok, detail) => results.push({ name, ok: !!ok, ...(detail !== undefined ? { detail } : {}) });

const pageFile = (id) => path.join(dist, id, 'index.html');
const docsHome = cfg.tabs[0].home;
const intHome = cfg.tabs[1].home;
const docsHtml = read(pageFile(docsHome));
const intHtml = read(pageFile(intHome));

// --- sidebar per tab -------------------------------------------------------
const sidebarOf = (h) => {
	const m = h.match(/<nav[^>]*aria-label="Main"[^>]*>([\s\S]*?)<\/nav>/) ?? h.match(/id="starlight__sidebar"[\s\S]*?<\/nav>/);
	const block = m ? m[0] : '';
	return [...block.matchAll(/href="([^"]+)"/g)].map((x) => x[1]);
};
const sbDocs = sidebarOf(docsHtml);
const sbInt = sidebarOf(intHtml);
check('sidebar: docs page lists only /docs/ links', sbDocs.length > 0 && sbDocs.every((h) => h.startsWith(`${base}/docs/`)), { links: sbDocs.length, sample: sbDocs.slice(0, 3) });
check('sidebar: internals page lists only /internals/ links', sbInt.length > 0 && sbInt.every((h) => h.startsWith(`${base}/internals/`)), { links: sbInt.length, sample: sbInt.slice(0, 3) });
check('sidebar: current page marked', /aria-current="page"[^>]*>|<a[^>]*aria-current="page"/.test(intHtml));

// --- tabs --------------------------------------------------------------------
const tabs = [...intHtml.matchAll(/<a href="([^"]+)"[^>]*data-tab="([^"]+)"/g)].map((m) => ({ href: m[1], tab: m[2], active: false }));
const active = intHtml.match(/<a href="[^"]+" aria-current="page" data-tab="([^"]+)"/)?.[1];
check('tabs: hrefs carry base', tabs.length === cfg.tabs.length && tabs.every((t) => t.href.startsWith(base + '/') && t.href.endsWith('/')), tabs);
check('tabs: internals tab active on internals page', active === 'internals', { active });
check('tabs: docs tab active on docs page', /aria-current="page" data-tab="docs"/.test(docsHtml));

// --- links -------------------------------------------------------------------
let unprefixed = 0;
let prefixed = 0;
let jsLinks = 0;
for (const f of scan) {
	const h = read(f);
	unprefixed += (h.match(/href="\/(docs|internals)\//g) ?? []).length;
	prefixed += (h.match(new RegExp(`href="${base}/(docs|internals)/`, 'g')) ?? []).length;
}
check('links: root-relative section links all carry base', base === '' || (unprefixed === 0 && prefixed > 0), { prefixed, unprefixed });
// DOM-level: executable <script> with fragment content, and javascript: URLs in real attributes
// (the same strings inside text or inside data-code="..." attribute values are inert).
let scriptAlert = 0;
for (const f of scan) {
	walkSync(parse(read(f)), (n) => {
		if (n.type !== ELEMENT_NODE) return;
		if (n.name === 'script' && n.children.some((c) => /alert\(/.test(c.value ?? ''))) scriptAlert++;
		for (const [k, v] of Object.entries(n.attributes ?? {})) {
			if ((k === 'href' || k === 'src' || k.startsWith('on')) && /^\s*javascript:/i.test(String(v))) jsLinks++;
		}
	});
}
check('links: no javascript: URLs in href/src attributes (DOM)', jsLinks === 0, { jsLinks });
check('hazard: no executable <script> containing fragment text (DOM)', scriptAlert === 0, { scriptAlert });

// --- hazards -----------------------------------------------------------------
const body = intHtml;
const lit = (s) => body.includes(s);
check('hazard: <script>alert(1)</script> rendered as text', lit('&lt;script&gt;alert(1)&lt;/script&gt;'));
check('hazard: Vec<String> literal', lit('Vec&lt;String&gt;'));
check('hazard: <T> literal', lit('&lt;T&gt;'));
check('hazard: {x} literal', lit('{x}'));
check('hazard: std::vec::Vec literal', lit('std::vec::Vec'));
check('hazard: a:b literal', lit('a:b'));
check('hazard: <!-- --> literal', lit('&lt;!-- hidden --&gt;'));
check('hazard: no smart punctuation (--json, quotes, ...)', lit('rfx query --json -- ') && !/[\u2013\u2014\u2018\u2019\u201c\u201d\u2026]/.test(body.match(/CLI flags like[^<]*/)?.[0] ?? '\u2014'));
check('hazard: :directive literal', lit(':directive') && lit('10:30'), { leaf: lit('::leaf') });
check('hazard: relative image not emitted as <img>', !/<img[^>]*src="x\.png"/.test(body) && lit('[image: diagram]'));
check('hazard: raw <a href=javascript> neutralised', !/<a href="javascript/i.test(body));
// With --prerender bundles (prose as inline HTML) highlighting is the producer's job, so
// expressive-code markup is only expected when Markdown fragments exist.
const hasMdFragments = (() => {
	const d = path.join(site, 'bundle', 'fragments');
	if (!fs.existsSync(d)) return false;
	const stack = [d];
	while (stack.length) {
		for (const e of fs.readdirSync(stack.pop(), { withFileTypes: true })) {
			if (e.isDirectory()) stack.push(path.join(e.parentPath, e.name));
			else if (e.name.endsWith('.md')) return true;
		}
	}
	return false;
})();
if (hasMdFragments) check('hazard: fenced rust code highlighted', /data-language="rust"/.test(body));
else check('prerendered: fenced rust code kept as language-tagged <pre>', /<code class="language-rust">/.test(body));
check('hazard: GFM table rendered', /<table>[\s\S]*Vec&lt;u32&gt;/.test(body));

// --- external requests -------------------------------------------------------
const ext = [];
for (const f of allScan) {
	if (!/\.(html|js|css|json|xml)$/.test(f)) continue;
	const t = read(f);
	for (const m of t.matchAll(/https:\/\/cdn|jsdelivr|unpkg|googleapis/g)) ext.push(`${rel(f)}: ${t.slice(Math.max(0, m.index - 40), m.index + 60).replace(/\s+/g, ' ')}`);
}
check('no CDN/googleapis references in dist', ext.length === 0, ext.slice(0, 5));

// --- TOC -----------------------------------------------------------------------
const toc = intHtml.match(/<starlight-toc[\s\S]*?<\/starlight-toc>/)?.[0] ?? '';
const tocText = toc.replace(/<[^>]+>/g, ' ');
check('toc: fragment headings present (Overview, Hazards, Safety)', ['Overview', 'Hazards', 'Safety'].every((t) => tocText.includes(t)));
check('toc: symbol headings present', /href="#sym-/.test(toc));
check('toc: symbol-doc headings excluded (toc:false)', !/-errors"/.test(toc));

// --- theme -------------------------------------------------------------------
check('theme: light/dark switch present', /<starlight-theme-select/.test(intHtml));
check('expressive-code: dual themes emitted', (() => {
	const css = all.filter((f) => /ec\..*\.css$/.test(f)).map(read).join('');
	return /data-theme='light'|data-theme="light"|\[data-theme=light\]/.test(css) && css.length > 0;
})());

// --- pagefind ------------------------------------------------------------------
const pfEntry = path.join(dist, 'pagefind', 'pagefind-entry.json');
const pf = fs.existsSync(pfEntry) ? JSON.parse(read(pfEntry)) : null;
const pfPages = pf ? Object.values(pf.languages ?? {}).reduce((a, l) => a + (l.page_count ?? 0), 0) : 0;
const bodyMarked = scan.filter((f) => /data-pagefind-body/.test(read(f))).length;
check('pagefind: index covers custom StarlightPage routes', pfPages >= cfg.generator.pages, { pfPages, pages: cfg.generator.pages, htmlWithPagefindBody: bodyMarked, scanned: scan.length });

// --- JS per page -----------------------------------------------------------------
// Static import closure of every module script a page loads, excluding dynamic imports.
const jsCache = new Map();
function closure(file, seen = new Set()) {
	if (seen.has(file) || !fs.existsSync(file)) return seen;
	seen.add(file);
	const src = read(file);
	for (const m of src.matchAll(/(?:^|[;\n}])\s*import\s*(?:[\w*{}\s,$]+from\s*)?["']([^"']+)["']/g)) {
		closure(path.resolve(path.dirname(file), m[1]), seen);
	}
	for (const m of src.matchAll(/\bfrom\s*["'](\.\/[^"']+)["']/g)) closure(path.resolve(path.dirname(file), m[1]), seen);
	return seen;
}
function pageJs(h) {
	const files = new Set();
	let inline = 0;
	for (const m of h.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)) {
		const src = m[1].match(/src="([^"]+)"/)?.[1];
		if (src) {
			const f = path.join(dist, src.replace(base, ''));
			for (const x of closure(f)) files.add(x);
		} else if (!/type="application\/(ld\+)?json"/.test(m[1])) {
			inline += Buffer.byteLength(m[2]);
			for (const im of m[2].matchAll(/import\s*["']([^"']+)["']|from\s*["']([^"']+)["']/g)) {
				const u = im[1] ?? im[2];
				if (u.startsWith(base + '/')) for (const x of closure(path.join(dist, u.slice(base.length)))) files.add(x);
			}
		}
	}
	for (const m of h.matchAll(/<link rel="modulepreload" href="([^"]+)"/g)) files.add(path.join(dist, m[1].replace(base, '')));
	const external = [...files].reduce((a, f) => a + (fs.existsSync(f) ? size(f) : 0), 0);
	return { inline, external, files: [...files].map(rel) };
}
const withMermaid = scan.filter((f) => /class="pulse-mermaid"/.test(read(f)));
const withoutMermaid = scan.filter((f) => !/class="pulse-mermaid"/.test(read(f)) && f.includes('/internals/'));
const jsA = pageJs(read(withoutMermaid[0]));
const jsB = withMermaid.length ? pageJs(read(withMermaid[0])) : null;
const mermaidStatic = [jsA, jsB].filter(Boolean).some((j) => j.files.some((f) => /mermaid/i.test(f)));
check('mermaid: not in the static JS of any page', !mermaidStatic, { sampleFiles: jsA.files });
check('JS per page <= 100 KB (excl. lazy mermaid)', jsA.inline + jsA.external <= 100 * 1024, { inline: jsA.inline, external: jsA.external });

// --- sizes -------------------------------------------------------------------------
const sum = (fs_) => fs_.reduce((a, f) => a + size(f), 0);
const htmlBytes = sum(html);
const pagefindBytes = sum(all.filter((f) => rel(f).startsWith('pagefind/')));
const astroBytes = sum(all.filter((f) => rel(f).startsWith('_astro/')));
const report = {
	site,
	pages: cfg.generator.pages,
	htmlFiles: html.length,
	distBytes: sum(all),
	distFiles: all.length,
	htmlBytes,
	avgHtmlBytes: Math.round(htmlBytes / html.length),
	astroAssetBytes: astroBytes,
	pagefindBytes,
	pagefindFiles: all.filter((f) => rel(f).startsWith('pagefind/')).length,
	jsPerPage: { inline: jsA.inline, external: jsA.external, total: jsA.inline + jsA.external },
	pagesWithMermaid: withMermaid.length * step,
	checks: results,
	failed: results.filter((r) => !r.ok).map((r) => r.name),
};
console.log(JSON.stringify(report, null, 1));
if (report.failed.length && !process.argv.includes('--sizes-only')) process.exit(1);
