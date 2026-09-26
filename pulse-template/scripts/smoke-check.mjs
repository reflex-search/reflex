#!/usr/bin/env node
// Checks on a built fixtures/smoke site (the runtime tarball smoke test).
//
//   node scripts/smoke-check.mjs <site_dir>      (reads <site_dir>/dist + pulse.config.json)
//
// Exits 1 when a check fails.
import fs from "node:fs";
import path from "node:path";

const site = path.resolve(process.argv[2] ?? ".");
const dist = path.join(site, "dist");
const cfg = JSON.parse(fs.readFileSync(path.join(site, "pulse.config.json"), "utf8"));
const base = cfg.base.replace(/\/$/, "");

const files = [];
(function walk(d) {
	for (const e of fs.readdirSync(d, { withFileTypes: true })) {
		const p = path.join(d, e.name);
		if (e.isDirectory()) walk(p);
		else files.push(p);
	}
})(dist);
const read = (p) => fs.readFileSync(p, "utf8");
const html = files.filter((f) => f.endsWith(".html"));
const page = (slug) => path.join(dist, slug, "index.html");

let failed = 0;
const check = (name, ok, detail) => {
	console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail === undefined ? "" : ` ${JSON.stringify(detail)}`}`);
	if (!ok) failed++;
};

check("home page", fs.existsSync(page("")));
const type = page("docs/reference/tally/tally");
check("type page has its methods", fs.existsSync(type) && read(type).includes("from_text"));
check("internals tab", fs.existsSync(page("internals")) && /data-tab="internals"/.test(read(page("internals"))));
check("pagefind index", fs.existsSync(path.join(dist, "pagefind", "pagefind-entry.json")));

let unprefixed = 0;
for (const f of html) unprefixed += (read(f).match(/href="\/(docs|internals)\//g) ?? []).length;
check("every section link carries the base path", base === "" || unprefixed === 0, { unprefixed });

const ext = [];
for (const f of files.filter((f) => /\.(html|js|css|json)$/.test(f))) {
	if (/https:\/\/cdn|jsdelivr|unpkg|googleapis/.test(read(f))) ext.push(path.relative(dist, f));
}
check("no CDN requests", ext.length === 0, ext.slice(0, 5));

console.log(`${html.length} HTML pages`);
process.exit(failed ? 1 : 0);
