#!/usr/bin/env node
/**
 * Cross-platform build for the integer GPT WebAssembly lanes.
 *
 * This replaces the Linux-only shell scripts with one entry point that works in
 * PowerShell, cmd, Git Bash, and CI. It builds the scalar module and the SIMD
 * module, then optionally copies them next to the TypeScript package.
 *
 * Usage:
 *   node scripts/build-wasm.mjs
 *   node scripts/build-wasm.mjs --build-dir dist-lib --build-type Release \
 *     --copy-wasm packages/only-gates-ai/wasm
 *
 * Environment variables used by the old scripts are still honoured:
 *   BUILD_DIR, BUILD_TYPE
 */
import { spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const isWindows = process.platform === "win32";

function readOption(name) {
  const index = process.argv.indexOf(name);
  if (index < 0) return undefined;
  const value = process.argv[index + 1];
  if (value === undefined || value.startsWith("--")) {
    fail(`${name} needs a value`);
  }
  return value;
}

function fail(message) {
  console.error(`build-wasm: ${message}`);
  process.exit(1);
}

const buildDirName =
  readOption("--build-dir") ?? process.env.BUILD_DIR ?? "dist";
const buildType =
  readOption("--build-type") ?? process.env.BUILD_TYPE ?? "Debug";
const copyWasm = readOption("--copy-wasm");
const buildDir = resolve(repoRoot, buildDirName);

function run(command, args, options = {}) {
  // cmd.exe is needed to resolve the .bat shims on Windows. Quote arguments
  // that contain spaces so a checkout under "C:\Program Files" still works.
  const shell = options.shell ?? false;
  const finalArgs = shell
    ? args.map((arg) => (/\s/.test(arg) ? `"${arg}"` : arg))
    : args;
  const result = spawnSync(command, finalArgs, {
    cwd: repoRoot,
    stdio: "inherit",
    shell,
  });
  if (result.error) {
    fail(`could not run ${command}: ${result.error.message}`);
  }
  if (result.status !== 0) {
    fail(`${command} exited with code ${result.status}`);
  }
}

function emcmakeCommand() {
  const emsdk = process.env.EMSDK;
  if (emsdk) {
    const shim = join(
      emsdk,
      "upstream",
      "emscripten",
      isWindows ? "emcmake.bat" : "emcmake",
    );
    if (existsSync(shim)) return { command: shim, shell: isWindows };
  }
  return {
    command: isWindows ? "emcmake.bat" : "emcmake",
    shell: isWindows,
  };
}

const configure = emcmakeCommand();
run(
  configure.command,
  [
    "cmake",
    "-S",
    repoRoot,
    "-B",
    buildDir,
    `-DCMAKE_BUILD_TYPE=${buildType}`,
  ],
  { shell: configure.shell },
);
run("cmake", ["--build", buildDir, "--parallel"]);

if (copyWasm) {
  const destination = resolve(repoRoot, copyWasm);
  mkdirSync(destination, { recursive: true });
  const files = [
    "only-gates-ai.js",
    "only-gates-ai.wasm",
    "only-gates-ai-simd.js",
    "only-gates-ai-simd.wasm",
  ];
  for (const file of files) {
    const source = join(buildDir, file);
    if (!existsSync(source)) {
      fail(`expected ${source}; did the Emscripten build succeed?`);
    }
    copyFileSync(source, join(destination, file));
  }
  console.log(`copied ${files.length} files to ${destination}`);
}

console.log(`built ${buildType} wasm lanes in ${buildDir}`);
