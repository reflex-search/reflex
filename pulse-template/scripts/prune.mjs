#!/usr/bin/env node
// Prune a production node_modules tree for the Pulse runtime tarball.
//
//   node scripts/prune.mjs <dir-containing-node_modules> [--dry-run]
//
// Removes files the build never reads. Every rule here was validated by
// building the fixture from the pruned tree (scripts/build-runtime.sh --verify).
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(process.argv[2] ?? '.');
const dry = process.argv.includes('--dry-run');
const nm = path.join(root, 'node_modules');
if (!fs.existsSync(nm)) {
	console.error(`prune: no node_modules under ${root}`);
	process.exit(2);
}

// Whole packages the static build never loads.
const DROP_PACKAGES = [
	'sharp', // passthroughImageService: no image transforms
	'@img', // sharp's libvips binaries
	'@types', // types only
	'typescript', // only for `astro check`, if it slipped in
];

// Package-root directories that are never runtime code.
const DROP_PKG_DIRS = new Set(['test', 'tests', '__tests__', 'docs', 'doc', 'demo', 'demos', 'example', 'examples', '.github', 'benchmark', 'benchmarks', 'coverage']);

// File patterns anywhere in the tree.
const DROP_FILE = [
	/\.map$/,
	/\.d\.[cm]?ts$/,
	/\.d\.ts\.map$/,
	/\.tsbuildinfo$/,
	/^(readme|changelog|history|contributing|code_of_conduct|security|authors|upgrading)(\.[a-z]+)?$/i,
	/\.(md|markdown)$/i, // LICENSE files are kept by the guard below
	/\.flow$/,
	/^\.(npmignore|eslintrc.*|prettierrc.*|editorconfig|travis\.yml|gitattributes)$/,
];
const KEEP_FILE = /^(licen[cs]e|notice|copying)(\.[a-z]+)?$/i;
// Astro and its integrations read .d.ts templates at build time (content-module-types.d.ts,
// client.d.ts for the generated .astro/types.d.ts), so declaration files stay in these trees.
const KEEP_DTS_UNDER = [`${path.sep}node_modules${path.sep}astro${path.sep}`, `${path.sep}node_modules${path.sep}@astrojs${path.sep}`];
const isDts = (name) => /\.d\.[cm]?ts$/.test(name);

// Package-specific: files not reachable from the package's ESM `import` export.
const DROP_PATHS = [
	// mermaid: Vite resolves `import('mermaid')` to dist/mermaid.core.mjs + dist/chunks/mermaid.core.
	'mermaid/dist/mermaid.js',
	'mermaid/dist/mermaid.min.js',
	'mermaid/dist/mermaid.esm.mjs',
	'mermaid/dist/mermaid.esm.min.mjs',
	'mermaid/dist/chunks/mermaid.esm',
	'mermaid/dist/chunks/mermaid.esm.min',
];

// pagefind_extended (CJK segmentation dictionaries) is 50 MB of the ~87 MB tarball.
// build-runtime.sh drops it and installs the 5 MB plain `pagefind` binary from the
// matching GitHub release instead; pagefind's resolver falls back to it
// (resolveBinaryPath(["pagefind_extended", "pagefind"])). --keep-pagefind-extended opts out.
if (!process.argv.includes('--keep-pagefind-extended')) {
	for (const e of fs.existsSync(path.join(nm, '@pagefind')) ? fs.readdirSync(path.join(nm, '@pagefind')) : []) {
		DROP_PATHS.push(`@pagefind/${e}/bin/pagefind_extended`, `@pagefind/${e}/bin/pagefind_extended.exe`);
	}
}

let files = 0;
let bytes = 0;
function rm(p) {
	let st;
	try {
		st = fs.lstatSync(p);
	} catch {
		return;
	}
	if (st.isDirectory()) {
		for (const e of fs.readdirSync(p)) rm(path.join(p, e));
		if (!dry) fs.rmdirSync(p);
	} else {
		files++;
		bytes += st.size;
		if (!dry) fs.unlinkSync(p);
	}
}

for (const pkg of DROP_PACKAGES) rm(path.join(nm, pkg));
for (const p of DROP_PATHS) rm(path.join(nm, p));

function isPackageRoot(dir) {
	return fs.existsSync(path.join(dir, 'package.json'));
}

(function walk(dir) {
	for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
		const p = path.join(dir, e.name);
		if (e.isSymbolicLink()) {
			// .bin shims (and any other links) are useless in a relocated tarball.
			rm(p);
			continue;
		}
		if (e.isDirectory()) {
			if (e.name === '.bin') {
				rm(p);
				continue;
			}
			if (DROP_PKG_DIRS.has(e.name) && isPackageRoot(dir)) {
				rm(p);
				continue;
			}
			walk(p);
		} else if (isDts(e.name) && KEEP_DTS_UNDER.some((k) => p.includes(k))) {
			continue;
		} else if (!KEEP_FILE.test(e.name) && DROP_FILE.some((re) => re.test(e.name))) {
			rm(p);
		}
	}
})(nm);

console.log(`prune: ${dry ? 'would remove' : 'removed'} ${files} files, ${(bytes / 1048576).toFixed(1)} MiB`);
