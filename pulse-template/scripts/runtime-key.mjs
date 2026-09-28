#!/usr/bin/env node
// Print the runtime id (DEPS_HASH) exactly as build.rs computes it:
// the first 12 hex digits of sha256("package.json" ‖ bytes ‖ "package-lock.json" ‖ bytes).
// The prebuilt runtime tarballs and the release tag `pulse-runtime-<id>` are named by it.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const h = createHash("sha256");
for (const name of ["package-lock.json", "package.json"].sort()) {
	h.update(name);
	h.update(readFileSync(join(root, name)));
}
console.log(h.digest("hex").slice(0, 12));
