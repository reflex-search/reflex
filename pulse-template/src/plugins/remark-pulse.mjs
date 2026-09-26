// Pulse Markdown sanitiser/rewriter for prose fragments.
//
// Astro 7 renders Markdown with Sätteri (native parser, JS visitors) by default,
// so this is a Sätteri *mdast* plugin — the Sätteri equivalent of a remark
// plugin. (Name kept as remark-pulse for the plan's vocabulary.)
//
// Fragments come from doc comments in arbitrary repositories, so they are
// untrusted text:
//   * raw HTML nodes become literal text (tiny allowlist of inert tags),
//   * javascript:/vbscript:/data: links are unwrapped to their text,
//   * relative images are not handed to Astro's image pipeline (the file does
//     not exist in the staged site) — they become literal text,
//   * root-relative /docs/ and /internals/ links get the site `base` prefix,
//   * ```mermaid fences become <pre class="pulse-mermaid"> (rendered lazily),
//   * fences without a language get the fragment's `lang` (rustdoc semantics).
//   * `depth_offset` frontmatter shifts heading levels (symbol docs start at #).
// Plain object: `defineMdastPlugin` from satteri is an identity helper for types only.

const ALLOWED_HTML = new Set(['<br>', '<br/>', '<br />', '<kbd>', '</kbd>', '<sub>', '</sub>', '<sup>', '</sup>']);
const UNSAFE_URL = /^\s*(javascript|vbscript|data|file):/i;
const HAS_SCHEME = /^[a-z][a-z0-9+.-]*:/i;

function escapeHtml(s) {
	return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

export function pulseMdastPlugin({ base = '/', sectionPrefixes = ['/docs/', '/internals/'] } = {}) {
	const b = base.endsWith('/') ? base.slice(0, -1) : base;
	const needsBase = (url) => b !== '' && sectionPrefixes.some((p) => url === p.slice(0, -1) || url.startsWith(p));

	return {
		name: 'pulse',
		html(node) {
			const v = node.value.trim().toLowerCase();
			if (ALLOWED_HTML.has(v)) return;
			return { type: 'text', value: node.value };
		},
		link(node, ctx) {
			if (UNSAFE_URL.test(node.url)) {
				ctx.replaceNode(node, [...node.children]);
				return;
			}
			if (needsBase(node.url)) ctx.setProperty(node, 'url', b + node.url);
		},
		definition(node, ctx) {
			if (UNSAFE_URL.test(node.url)) ctx.setProperty(node, 'url', '#');
			else if (needsBase(node.url)) ctx.setProperty(node, 'url', b + node.url);
		},
		image(node) {
			const url = node.url ?? '';
			if (UNSAFE_URL.test(url) || (!HAS_SCHEME.test(url) && !url.startsWith('/'))) {
				return { type: 'text', value: `[image: ${node.alt || url}]` };
			}
		},
		code(node, ctx) {
			if (node.lang === 'mermaid') {
				return { rawHtml: `<pre class="pulse-mermaid">${escapeHtml(node.value)}</pre>` };
			}
			if (!node.lang) {
				const lang = ctx.data?.astro?.frontmatter?.lang;
				ctx.setProperty(node, 'lang', typeof lang === 'string' && lang ? lang : 'text');
			}
		},
		heading(node, ctx) {
			const off = Number(ctx.data?.astro?.frontmatter?.depth_offset ?? 0);
			if (off > 0) ctx.setProperty(node, 'depth', Math.min(6, node.depth + off));
		},
	};
}
