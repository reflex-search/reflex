import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';
import pulse from '../pulse.config.json';

// build.pageSource = 'fs' (spike variant D): page JSON is read from disk at render time
// and never enters the content store (whose size grows the heap and data-store.json).
const pagesFromFs = (pulse as any).build?.pageSource === 'fs';

// Ids are the file path minus the extension, verbatim: `internals/core/mod-0/index`.
// The default glob id slugifies (lowercases, strips `_`), which would break the
// page ids Rust writes into block references.
const stemId = ({ entry }: { entry: string }) => entry.replace(/\.(json|md)$/, '');

export const collections = {
	// Starlight's own collection. Pulse writes nothing here; Starlight only needs it to exist.
	docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),

	// One JSON file per page: { title, description, tab, blocks: [...] }.
	// Blocks are validated loosely: the Rust side owns the schema, and a strict
	// per-block zod union costs real time at 20k pages.
	pages: defineCollection({
		loader: glob({ pattern: pagesFromFs ? '__none__/*.json' : '**/*.json', base: './bundle/pages', generateId: stemId, retainBody: false }),
		schema: z.object({
			title: z.string(),
			description: z.string().optional(),
			tab: z.string(),
			blocks: z.array(z.looseObject({ type: z.string() })),
		}),
	}),

	// Prose fragments (plain Markdown, never MDX). Rendered through the Sätteri
	// pipeline once at sync time and cached in the content store by digest.
	fragments: defineCollection({
		loader: glob({ pattern: '**/*.md', base: './bundle/fragments', generateId: stemId, retainBody: false }),
		schema: z.object({
			page: z.string(),
			toc: z.boolean().default(true),
			lang: z.string().default('text'),
			origin: z.string().optional(),
			depth_offset: z.number().int().min(0).max(5).optional(),
		}),
	}),
};
