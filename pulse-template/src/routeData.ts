// Starlight route middleware: one sidebar per tab, chosen by URL prefix.
//
// The global Starlight sidebar is empty (astro.config.mjs). For each page we
// build the sidebar of the page's tab from pulse.config.json. Large trees are
// pruned to the current branch: rendering a 20k-link sidebar into each of 20k
// pages is O(N^2) bytes (tens of GB), so beyond `FULL_TREE_MAX` links the
// groups off the current path collapse to a single link to their first page.
import { defineRouteMiddleware } from '@astrojs/starlight/route-data';
import pulse from '../pulse.config.json';

type Node = { label: string; slug?: string; items?: Node[] };
type Link = { type: 'link'; label: string; href: string; isCurrent: boolean; badge: undefined; attrs: Record<string, never> };
type Group = { type: 'group'; label: string; entries: Entry[]; collapsed: boolean; badge: undefined };
type Entry = Link | Group;

const FULL_TREE_MAX: number = (pulse as any).sidebarFullTreeMax ?? 800;
const base: string = (pulse.base ?? '/').replace(/\/$/, '');
const href = (slug: string) => (slug ? `${base}/${slug}/` : `${base}/`);

interface TabIndex {
	tree: Node[];
	order: { slug: string; label: string }[];
	position: Map<string, number>;
	/** Node chain (groups) containing each slug, outermost first. */
	ancestors: Map<string, Node[]>;
	full: boolean;
}

const tabs = new Map<string, TabIndex>();
for (const tab of pulse.tabs) {
	const tree: Node[] = (pulse.sidebar as Record<string, Node[]>)[tab.id] ?? [];
	const order: { slug: string; label: string }[] = [];
	const ancestors = new Map<string, Node[]>();
	const walk = (nodes: Node[], chain: Node[]) => {
		for (const n of nodes) {
			if (n.slug) {
				order.push({ slug: n.slug, label: n.label });
				ancestors.set(n.slug, chain);
			}
			if (n.items) walk(n.items, [...chain, n]);
		}
	};
	walk(tree, []);
	tabs.set(tab.id, {
		tree,
		order,
		position: new Map(order.map((o, i) => [o.slug, i])),
		ancestors,
		full: order.length <= FULL_TREE_MAX,
	});
}

function firstSlug(n: Node): string | undefined {
	if (n.slug) return n.slug;
	for (const c of n.items ?? []) {
		const s = firstSlug(c);
		if (s) return s;
	}
}

function build(nodes: Node[], current: string, onPath: Set<Node>, full: boolean): Entry[] {
	const out: Entry[] = [];
	for (const n of nodes) {
		if (n.items) {
			if (full || onPath.has(n)) {
				out.push({ type: 'group', label: n.label, entries: build(n.items, current, onPath, full), collapsed: !onPath.has(n), badge: undefined });
			} else {
				const s = firstSlug(n);
				if (s) out.push(link(`${n.label} ›`, s, false));
			}
		} else if (n.slug) {
			out.push(link(n.label, n.slug, n.slug === current));
		}
	}
	return out;
}

function link(label: string, slug: string, isCurrent: boolean): Link {
	return { type: 'link', label, href: href(slug), isCurrent, badge: undefined, attrs: {} };
}

/** URL pathname -> [tab, slug] where slug is the page id (`internals/core/mod-0/index`). */
export function locate(pathname: string): { tab: string; slug: string } | undefined {
	let p = pathname.startsWith(base + '/') ? pathname.slice(base.length) : pathname;
	p = p.replace(/^\/+|\/+$/g, '');
	if (p === '') return { tab: 'docs', slug: '' };
	for (const tab of pulse.tabs) {
		const prefix = tab.prefix.replace(/^\/+|\/+$/g, '');
		if (p === prefix || p.startsWith(prefix + '/')) return { tab: tab.id, slug: p };
	}
}

export const onRequest = defineRouteMiddleware((context) => {
	const route = context.locals.starlightRoute;
	const where = locate(context.url.pathname);
	if (!where) return;
	const idx = tabs.get(where.tab);
	if (!idx) return;
	const onPath = new Set(idx.ancestors.get(where.slug) ?? []);
	route.sidebar = build(idx.tree, where.slug, onPath, idx.full) as typeof route.sidebar;

	// Starlight derived pagination from the (empty) global sidebar before we ran;
	// recompute it from the tab's full reading order, which pruning does not change.
	const i = idx.position.get(where.slug);
	if (i !== undefined) {
		const prev = idx.order[i - 1];
		const next = idx.order[i + 1];
		route.pagination = {
			prev: prev ? link(prev.label, prev.slug, false) : undefined,
			next: next ? link(next.label, next.slug, false) : undefined,
		} as typeof route.pagination;
	}
});
