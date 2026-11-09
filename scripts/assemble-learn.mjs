#!/usr/bin/env node
/*
 * Copy the prerendered learn site into the main app's output. The Pages
 * workflow publishes apps/only-gates-ai/dist, so /learn/ lives beside the SPA
 * without changing the deploy directory or the SPA's routing.
 */

import { cp, mkdir, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const source = join(root, "apps", "only-gates-ai-learn", "dist");
const target = join(root, "apps", "only-gates-ai", "dist", "learn");

await rm(target, { recursive: true, force: true });
await mkdir(target, { recursive: true });
await cp(source, target, { recursive: true });
console.log("assembled learn site at apps/only-gates-ai/dist/learn");
