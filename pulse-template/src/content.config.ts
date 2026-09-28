import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

// Pulse pages never enter the content store: Rust renders them to HTML and the page
// routes read bundle/pages/<id>.json at render time (spike design D). Starlight only
// needs its own collection to exist.
export const collections = {
	docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
};
