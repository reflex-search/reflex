#!/usr/bin/env node
// Synthetic Pulse page bundle generator (M0a renderer spike).
//
// Writes what the Rust side of `rfx pulse` is expected to write into a staged
// Astro project:
//
//   <out>/pulse.config.json                      site + tabs + nested sidebar per tab
//   <out>/bundle/pages/<tab>/<path>.json         one page = ordered list of blocks
//   <out>/bundle/fragments/<tab>/<path>/<n>.md   prose fragments (plain Markdown, never MDX)
//
// Usage:
//   node scripts/gen-fixture.mjs --out <dir> --pages N [--symbols-per-page K]
//        [--mermaid-every M] [--base /reflex/] [--site https://example.org]
//        [--prerender] [--plain-signatures]
//        (--prerender: experiment E2, prose is rendered to sanitised HTML here, the way
//         Rust would with pulldown-cmark, and inlined into the page JSON; no .md files.
//         --plain-signatures: signatures as plain <pre>, i.e. highlighted outside Astro.)
//   node scripts/gen-fixture.mjs --out <dir> --touch 0.01 [--seed 2]
//        (rewrite ~1% of the existing fragments in place, for warm-build tests)
//
// Deterministic: same arguments -> byte-identical output.

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const args = parseArgs(process.argv.slice(2));
const out = path.resolve(args.out ?? 'fixture');

if (args.touch !== undefined) {
	touchFragments(out, Number(args.touch), Number(args.seed ?? 2));
	process.exit(0);
}

const N = Number(args.pages ?? 100);
const K = Number(args['symbols-per-page'] ?? 8);
const MERMAID_EVERY = Number(args['mermaid-every'] ?? 20);
const BASE = normBase(args.base ?? '/');
const SITE = args.site ?? 'https://example.org';
// Optional astro build knobs (see astro.config.mjs), e.g. --build '{"chunkedStore":true}'.
const BUILD = args.build ? JSON.parse(args.build) : undefined;
const PRERENDER = args.prerender === 'true';
const PLAIN_SIG = args['plain-signatures'] === 'true';
const render = PRERENDER ? await prerenderer() : null;

const rng = mulberry32(0x5eed ^ N);
const pick = (a) => a[Math.floor(rng() * a.length)];

// ---------------------------------------------------------------- vocabulary
const VERBS = ['Builds', 'Parses', 'Resolves', 'Merges', 'Extracts', 'Validates', 'Streams', 'Caches', 'Normalises', 'Intersects', 'Flushes', 'Compacts'];
const NOUNS = ['trigram posting list', 'symbol table', 'content store', 'path map', 'freshness verdict', 'dependency graph', 'import edge', 'query plan', 'line index', 'shard header', 'batch writer', 'glob matcher'];
const ADJ = ['sorted', 'memory-mapped', 'compressed', 'incremental', 'deterministic', 'lazily parsed', 'per-shard', 'deduplicated'];
const KINDS = ['Function', 'Struct', 'Enum', 'Trait', 'Method', 'Constant', 'TypeAlias', 'Macro'];
const TOPICS = ['core', 'storage', 'query', 'parsers', 'indexer', 'mcp', 'cli', 'deps', 'watcher', 'semantic', 'pulse', 'serve', 'cache', 'config', 'output', 'trigram', 'freshness', 'symbols', 'ast', 'glob', 'analyze', 'context', 'ranking', 'streams', 'shards', 'paths', 'tiers', 'schema', 'bench', 'tests'];
const DOC_SECTIONS = ['getting-started', 'guides', 'reference', 'concepts', 'recipes', 'faq'];

// Every hazard the renderer must print literally (or neutralise).
const HAZARD_PARAGRAPH = [
	'It returns a Vec<String> and is generic over <T> where T: AsRef<str>; see std::vec::Vec and the pair a:b.',
	'Template-looking braces such as {x} and {{y}} are plain text, not expressions.',
	'A comment <!-- hidden --> must stay visible, and <script>alert(1)</script> must never run.',
	'A [bad link](javascript:alert(1)) is neutralised; so is <a href="javascript:alert(2)">raw</a>.',
	'Directive-looking text :directive and ::leaf{#id} and a time like 10:30 stay literal.',
	'Relative images like ![diagram](x.png) are not resolved by the image pipeline.',
	'CLI flags like rfx query --json -- "x" and \'quotes\' and ... stay ASCII.',
].join('\n');

// ---------------------------------------------------------------- structure
const docsCount = Math.max(5, Math.round(N * 0.05));
const internalsCount = Math.max(1, N - docsCount);
const T = Math.max(2, Math.round(Math.cbrt(internalsCount)));
const M = T;
const perModule = Math.ceil(internalsCount / (T * M));

// Only ever remove what this script owns: <out> may be a staged site dir.
fs.rmSync(path.join(out, 'bundle'), { recursive: true, force: true });
fs.rmSync(path.join(out, 'pulse.config.json'), { force: true });
const pagesDir = path.join(out, 'bundle', 'pages');
const fragDir = path.join(out, 'bundle', 'fragments');
fs.mkdirSync(pagesDir, { recursive: true });
fs.mkdirSync(fragDir, { recursive: true });

const pageIds = [];
const indexRows = [];
let pageIndex = 0;
let fragCount = 0;

// docs tab ----------------------------------------------------------------
const docsSidebar = [];
{
	const perSection = Math.ceil(docsCount / DOC_SECTIONS.length);
	let made = 0;
	for (const section of DOC_SECTIONS) {
		if (made >= docsCount) break;
		const group = { label: title(section), items: [] };
		for (let i = 0; i < perSection && made < docsCount; i++, made++) {
			const slug = made === 0 ? 'intro' : `${section}/page-${i}`;
			const id = `docs/${slug}`;
			group.items.push({ label: made === 0 ? 'Introduction' : `${title(section)} ${i}`, slug: id });
			writeDocsPage(id, made === 0 ? 'Introduction' : `${title(section)} ${i}`);
		}
		docsSidebar.push(group);
	}
}

// internals tab -----------------------------------------------------------
const internalsSidebar = [];
{
	let made = 0;
	outer: for (let t = 0; t < T; t++) {
		const topic = TOPICS[t % TOPICS.length] + (t >= TOPICS.length ? `-${Math.floor(t / TOPICS.length)}` : '');
		const topicGroup = { label: title(topic), items: [] };
		internalsSidebar.push(topicGroup);
		for (let m = 0; m < M; m++) {
			const mod = `mod-${m}`;
			const modGroup = { label: `${topic}::${mod}`, items: [] };
			topicGroup.items.push(modGroup);
			for (let p = 0; p < perModule; p++) {
				if (made >= internalsCount) break outer;
				const id = `internals/${topic}/${mod}/${p === 0 ? 'index' : `item-${p}`}`;
				const label = p === 0 ? `${mod} (overview)` : `${topic}::${mod}::item_${p}`;
				modGroup.items.push({ label, slug: id });
				writeInternalsPage(id, label, topic, mod, p);
				made++;
			}
		}
	}
}

const config = {
	title: 'Reflex',
	description: 'Synthetic Pulse fixture',
	site: SITE,
	base: BASE,
	tabs: [
		{ id: 'docs', label: 'Docs', prefix: '/docs/', home: 'docs/intro' },
		{ id: 'internals', label: 'Internals', prefix: '/internals/', home: firstSlug(internalsSidebar) },
	],
	sidebar: { docs: docsSidebar, internals: internalsSidebar },
	defaultLang: 'rust',
	...(BUILD ? { build: BUILD } : {}),
	generator: { pages: pageIds.length, symbolsPerPage: K, mermaidEvery: MERMAID_EVERY, fragments: fragCount },
};
fs.writeFileSync(path.join(out, 'pulse.config.json'), JSON.stringify(config, null, 1));
// Page index for build.pageSource = 'fs' (with md fragments the hash would have to cover them too).
fs.writeFileSync(path.join(out, 'bundle', 'index.json'), JSON.stringify(indexRows));
console.log(`fixture: ${pageIds.length} pages (${docsCount} docs, ${pageIds.length - docsCount} internals), ${fragCount} fragments, T=${T} M=${M} perModule=${perModule} -> ${out}`);

// ---------------------------------------------------------------- writers
function writeDocsPage(id, pageTitle) {
	const blocks = [];
	blocks.push({ type: 'prose', ...prose(id, 0, docsProse(id)) });
	blocks.push({ type: 'heading', level: 2, text: 'Configuration', id: 'configuration' });
	blocks.push({
		type: 'table',
		columns: ['Key', 'Default', 'Meaning'],
		rows: [
			['index.max_file_size', '10485760', 'Files larger than this are skipped'],
			['search.default_limit', '100', 'Default result cap (<T> is literal here)'],
			['performance.parallel_threads', '0', 'Auto: 80% of cores {x}'],
		],
	});
	blocks.push({ type: 'prose', ...prose(id, 1, docsProse(id, true)) });
	writePage(id, { title: pageTitle, description: `${pageTitle} — Reflex documentation`, tab: 'docs', blocks });
}

function writeInternalsPage(id, label, topic, mod, p) {
	const blocks = [];
	const lang = 'rust';
	blocks.push({ type: 'prose', ...prose(id, 0, moduleProse(id, topic, mod), { lang, origin: `src/${topic}/${mod}.rs:1` }) });
	if (pageIndex % MERMAID_EVERY === 0) {
		blocks.push({ type: 'heading', level: 2, text: 'Dependencies', id: 'dependencies' });
		blocks.push({ type: 'dep-graph', mermaid: depGraph(topic, mod) });
	}
	blocks.push({ type: 'heading', level: 2, text: 'Symbols', id: 'symbols' });
	const rows = [];
	for (let s = 0; s < K; s++) {
		const kind = KINDS[(s + p) % KINDS.length];
		const name = symbolName(kind, s);
		const line = 10 + s * 37;
		const signature = signatureFor(kind, name);
		rows.push([name, kind, `src/${topic}/${mod}.rs:${line}`]);
		blocks.push({
			type: 'symbol-card',
			id: `sym-${name.toLowerCase()}`,
			name,
			kind,
			path: `src/${topic}/${mod}.rs`,
			line,
			signature,
			lang,
			...docFields(prose(id, s + 1, symbolDoc(name, kind, topic, mod), { lang, origin: `src/${topic}/${mod}.rs:${line - 3}`, toc: false, depth_offset: 3 })),
			...(PLAIN_SIG ? { signature_html: `<pre class="pulse-sig"><code class="language-${lang}">${esc(signature)}</code></pre>` } : {}),
			refs: 1 + Math.floor(rng() * 40),
		});
	}
	blocks.push({ type: 'heading', level: 2, text: 'Summary', id: 'summary' });
	blocks.push({ type: 'table', columns: ['Symbol', 'Kind', 'Location'], rows });
	writePage(id, { title: label, description: `${topic}::${mod} internals`, tab: 'internals', blocks });
}

function writePage(id, page) {
	const file = path.join(pagesDir, `${id}.json`);
	fs.mkdirSync(path.dirname(file), { recursive: true });
	const json = JSON.stringify(page);
	fs.writeFileSync(file, json);
	indexRows.push({ id, tab: page.tab, title: page.title, hash: hash(json) });
	pageIds.push(id);
	pageIndex++;
}

/** Block fields for one piece of prose: a fragment reference, or inline HTML (--prerender). */
function prose(pageId, n, body, meta = {}) {
	if (!render) return { fragment: frag(pageId, n, body, meta) };
	fragCount++;
	return { ...render(body, meta), toc: meta.toc ?? true };
}
function docFields(p) {
	return p.fragment ? { doc: p.fragment } : { doc_html: p.html, doc_headings: p.headings };
}
function hash(s) {
	return crypto.createHash('sha1').update(s).digest('hex').slice(0, 16);
}
function esc(s) {
	return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
async function prerenderer() {
	const { markdownToHtml } = await import('satteri');
	const { pulseMdastPlugin } = await import('../src/plugins/remark-pulse.mjs');
	const plugin = pulseMdastPlugin({ base: BASE });
	return (body, meta) => {
		const { html } = markdownToHtml(body, {
			mdastPlugins: [plugin],
			features: { gfm: true, headingAttributes: true, smartPunctuation: false },
			data: { astro: { frontmatter: { lang: meta.lang ?? 'text', depth_offset: meta.depth_offset } } },
		});
		const headings = [];
		const seen = new Map();
		const out = html.replace(/<h([1-6])(?: id="([^"]*)")?>([\s\S]*?)<\/h\1>/g, (_, d, id, inner) => {
			const text = inner.replace(/<[^>]+>/g, '');
			let slug = id || text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
			const k = seen.get(slug) ?? 0;
			seen.set(slug, k + 1);
			if (k) slug += `-${k}`;
			headings.push({ depth: Number(d), slug, text });
			return `<h${d} id="${slug}">${inner}</h${d}>`;
		});
		return { html: out, headings };
	};
}

function frag(pageId, n, body, meta = {}) {
	const id = `${pageId}/${n}`;
	const file = path.join(fragDir, `${id}.md`);
	fs.mkdirSync(path.dirname(file), { recursive: true });
	const fm = [
		'---',
		`page: ${JSON.stringify(pageId)}`,
		`toc: ${meta.toc ?? true}`,
		`lang: ${JSON.stringify(meta.lang ?? 'text')}`,
		`origin: ${JSON.stringify(meta.origin ?? 'README.md:1')}`,
		...(meta.depth_offset ? [`depth_offset: ${meta.depth_offset}`] : []),
		'---',
		'',
	].join('\n');
	fs.writeFileSync(file, fm + body + '\n');
	fragCount++;
	return id;
}

// ---------------------------------------------------------------- content
function sentence() {
	return `${pick(VERBS)} the ${pick(ADJ)} ${pick(NOUNS)} for the ${pick(NOUNS)}, returning early when the ${pick(NOUNS)} is already ${pick(ADJ)}.`;
}
function para(n = 3) {
	return Array.from({ length: n }, sentence).join(' ');
}

function moduleProse(id, topic, mod) {
	const other = pageIds.length > 3 ? pageIds[Math.floor(rng() * pageIds.length)] : 'docs/intro';
	return [
		para(4),
		'',
		'## Overview',
		'',
		para(3),
		`See [the ${topic} overview](/internals/${topic}/mod-0/index/) and the [introduction](/docs/intro/), or [a related page](/${other}/).`,
		'',
		'### Hazards',
		'',
		HAZARD_PARAGRAPH,
		'',
		'```rust',
		`pub fn ${topic}_${mod.replace('-', '_')}(paths: Vec<String>) -> Result<Vec<u32>, Error> {`,
		'    let map: HashMap<String, Vec<u32>> = HashMap::new(); // <T> and {x} in code',
		'    Ok(paths.iter().map(|p| p.len() as u32).collect())',
		'}',
		'```',
		'',
		'```',
		'plain fence without a language: <script>alert(3)</script>',
		'```',
		'',
		'| Field | Type | Notes |',
		'| --- | --- | --- |',
		'| `ids` | `Vec<u32>` | sorted, deduplicated |',
		'| `map` | `HashMap<K, V>` | keyed by {path} |',
		'',
		(pageIndex % 50 === 1) ? ['```mermaid', 'graph TD', `  A[${topic}] --> B[${mod}]`, '  B --> C["Vec&lt;String&gt;"]', '```', ''].join('\n') : '',
		'### Safety',
		'',
		para(2),
	].join('\n');
}

function docsProse(id, second = false) {
	if (second) {
		return ['## Troubleshooting', '', para(3), '', '- ' + sentence(), '- ' + sentence(), '', HAZARD_PARAGRAPH].join('\n');
	}
	return [
		para(3),
		'',
		'## Quick start',
		'',
		'```sh',
		'rfx index && rfx query "Vec<String>" --json',
		'```',
		'',
		`Continue with [the internals](/internals/) or jump to [the reference](/docs/reference/page-0/).`,
		'',
		para(2),
	].join('\n');
}

function symbolDoc(name, kind, topic, mod) {
	const hid = `sym-${name.toLowerCase()}`;
	return [
		`${pick(VERBS)} the ${pick(NOUNS)} (${kind.toLowerCase()} \`${name}\`).`,
		'',
		para(2),
		'',
		`# Errors {#${hid}-errors}`,
		'',
		`Returns \`Err\` when the ${pick(NOUNS)} is ${pick(ADJ)}; generic over <T>, e.g. Vec<String>.`,
		'',
		`# Examples {#${hid}-examples}`,
		'',
		'```',
		`let out = ${name.toLowerCase()}(&input)?; // {x}`,
		'```',
	].join('\n');
}

function signatureFor(kind, name) {
	switch (kind) {
		case 'Function': return `pub fn ${snake(name)}<T: AsRef<str>>(items: &[T], limit: usize) -> Result<Vec<String>, Error>`;
		case 'Method': return `pub fn ${snake(name)}(&mut self, key: &str) -> Option<&Vec<u32>>`;
		case 'Struct': return `pub struct ${name}<'a> {\n    pub path: &'a str,\n    pub ids: Vec<u32>,\n}`;
		case 'Enum': return `pub enum ${name} {\n    Fresh,\n    Stale { changed: usize },\n}`;
		case 'Trait': return `pub trait ${name}: Send + Sync {\n    fn visit(&self, node: &Node) -> bool;\n}`;
		case 'Constant': return `pub const ${snake(name).toUpperCase()}: usize = 1 << 16;`;
		case 'TypeAlias': return `pub type ${name} = std::collections::HashMap<String, Vec<u32>>;`;
		default: return `macro_rules! ${snake(name)} { ($x:expr) => { $x } }`;
	}
}

function symbolName(kind, s) {
	const base = pick(NOUNS).split(/[\s-]/).map(title).join('');
	return kind === 'Function' || kind === 'Method' || kind === 'Macro' ? `${base}${s}` : `${base}${kind}${s}`;
}

function depGraph(topic, mod) {
	return [
		'graph LR',
		`  ${topic}_${mod.replace('-', '_')}["${topic}::${mod}"] --> storage["storage"]`,
		`  ${topic}_${mod.replace('-', '_')} --> query["query"]`,
		'  query --> trigram["trigram"]',
		'  storage --> trigram',
	].join('\n');
}

// ---------------------------------------------------------------- touch mode
function touchFragments(dir, fraction, seed) {
	const root = path.join(dir, 'bundle', 'fragments');
	const files = [];
	if (fs.existsSync(root)) (function walk(d) {
		for (const e of fs.readdirSync(d, { withFileTypes: true })) {
			const p = path.join(d, e.name);
			if (e.isDirectory()) walk(p);
			else if (e.name.endsWith('.md')) files.push(p);
		}
	})(root);
	files.sort();
	const r = mulberry32(seed);
	let n = 0;
	if (files.length === 0) {
		// --prerender bundles have no .md files: edit ~fraction of the prose blocks inside page JSON
		// (one page holds 1 + K prose pieces, so scale the per-page probability to match).
		const pages = [];
		(function walk(d) {
			for (const e of fs.readdirSync(d, { withFileTypes: true })) {
				const p = path.join(d, e.name);
				if (e.isDirectory()) walk(p);
				else if (e.name.endsWith('.json')) pages.push(p);
			}
		})(path.join(dir, 'bundle', 'pages'));
		pages.sort();
		let pieces = 0;
		for (const f of pages) {
			const page = JSON.parse(fs.readFileSync(f, 'utf8'));
			const prose = page.blocks.filter((b) => b.html !== undefined || b.doc_html !== undefined);
			pieces += prose.length;
			let hit = false;
			for (const b of prose) {
				if (r() < fraction) {
					if (b.html !== undefined) b.html += `<p>Edited (seed ${seed}): ${Date.now()}.</p>`;
					else b.doc_html += `<p>Edited (seed ${seed}): ${Date.now()}.</p>`;
					n++;
					hit = true;
				}
			}
			if (hit) fs.writeFileSync(f, JSON.stringify(page));
		}
		// Keep bundle/index.json hashes in step with the edited pages.
		const idxFile = path.join(dir, 'bundle', 'index.json');
		if (fs.existsSync(idxFile)) {
			const idx = JSON.parse(fs.readFileSync(idxFile, 'utf8'));
			for (const row of idx) row.hash = hash(fs.readFileSync(path.join(dir, 'bundle', 'pages', `${row.id}.json`), 'utf8'));
			fs.writeFileSync(idxFile, JSON.stringify(idx));
		}
		console.log(`touched ${n}/${pieces} inline prose pieces (${((100 * n) / pieces).toFixed(2)}%)`);
		return;
	}
	for (const f of files) {
		if (r() < fraction) {
			fs.appendFileSync(f, `\nEdited (seed ${seed}): ${Date.now()}.\n`);
			n++;
		}
	}
	console.log(`touched ${n}/${files.length} fragments (${((100 * n) / files.length).toFixed(2)}%)`);
}

// ---------------------------------------------------------------- helpers
function firstSlug(tree) {
	for (const n of tree) {
		if (n.slug) return n.slug;
		if (n.items) {
			const s = firstSlug(n.items);
			if (s) return s;
		}
	}
	return undefined;
}
function title(s) {
	return s.replace(/(^|[-_ ])(\w)/g, (_, a, c) => (a ? ' ' : '') + c.toUpperCase()).trim();
}
function snake(s) {
	return s.replace(/([a-z0-9])([A-Z])/g, '$1_$2').toLowerCase();
}
function normBase(b) {
	if (!b.startsWith('/')) b = '/' + b;
	if (!b.endsWith('/')) b += '/';
	return b;
}
function mulberry32(a) {
	return function () {
		a |= 0;
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}
function parseArgs(argv) {
	const o = {};
	for (let i = 0; i < argv.length; i++) {
		const a = argv[i];
		if (!a.startsWith('--')) continue;
		const k = a.slice(2);
		const v = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : 'true';
		o[k] = v;
	}
	return o;
}
