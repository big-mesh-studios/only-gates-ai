import JSZip from "jszip";

/*
 * Small ZIP wrapper used by the model bundle export/import path.
 *
 * The bundle is a flat archive of a handful of named files, so we only need
 * regular files at the archive root. Rejecting directories, nested paths, and
 * unknown compression methods keeps the importer from having to trust archive
 * metadata any more than necessary.
 */

const MAX_ENTRIES = 32;
const MAX_ENTRY_BYTES = 256 * 1024 * 1024;
const MAX_TOTAL_BYTES = 512 * 1024 * 1024;

export interface ZipFileInput {
  readonly name: string;
  readonly data: Uint8Array | string;
}

export type ZipSource = ArrayBuffer | Uint8Array | Blob;

function assertSafeName(name: string): void {
  if (
    name.length === 0 ||
    name.length > 128 ||
    name.includes("/") ||
    name.includes("\\") ||
    name.includes("..") ||
    name.startsWith(".") ||
    name !== name.trim()
  ) {
    throw new Error(`Unsafe file name in model bundle: ${JSON.stringify(name)}.`);
  }
}

/** Packs the supplied files into a standard deflate-compressed ZIP archive. */
export async function createZip(
  files: readonly ZipFileInput[],
): Promise<Uint8Array> {
  if (files.length > MAX_ENTRIES) {
    throw new Error(`Model bundles may contain at most ${MAX_ENTRIES} files.`);
  }
  const zip = new JSZip();
  const seen = new Set<string>();
  let total = 0;
  for (const file of files) {
    assertSafeName(file.name);
    if (seen.has(file.name)) {
      throw new Error(`Duplicate file in model bundle: ${file.name}.`);
    }
    seen.add(file.name);
    const size =
      typeof file.data === "string"
        ? new TextEncoder().encode(file.data).byteLength
        : file.data.byteLength;
    if (size > MAX_ENTRY_BYTES) {
      throw new Error(`Model bundle entry is too large: ${file.name}.`);
    }
    total += size;
    zip.file(file.name, file.data);
  }
  if (total > MAX_TOTAL_BYTES) {
    throw new Error(
      `Model bundle is too large (${total.toLocaleString()} bytes before compression).`,
    );
  }
  return zip.generateAsync({
    type: "uint8array",
    compression: "DEFLATE",
    compressionOptions: { level: 6 },
  });
}

/**
 * Reads a ZIP archive and returns its regular files by name.
 *
 * JSZip verifies each entry's CRC when `checkCRC32` is enabled. The extra size
 * checks below bound the in-memory result in case of a malicious archive.
 */
export async function readZip(
  source: ZipSource,
): Promise<Map<string, Uint8Array>> {
  const zip = await JSZip.loadAsync(source, { checkCRC32: true });
  const entries = Object.entries(zip.files).filter(([, entry]) => !entry.dir);
  if (entries.length > MAX_ENTRIES) {
    throw new Error(`Model bundles may contain at most ${MAX_ENTRIES} files.`);
  }

  const files = new Map<string, Uint8Array>();
  let total = 0;
  for (const [name, entry] of entries) {
    assertSafeName(name);
    const data = await entry.async("uint8array");
    if (data.byteLength > MAX_ENTRY_BYTES) {
      throw new Error(`Model bundle entry is too large: ${name}.`);
    }
    total += data.byteLength;
    if (total > MAX_TOTAL_BYTES) {
      throw new Error("Model bundle is too large when decompressed.");
    }
    files.set(name, data);
  }
  return files;
}
