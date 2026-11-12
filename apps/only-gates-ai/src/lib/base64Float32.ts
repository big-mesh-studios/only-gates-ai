/*
 * Base64 helpers for float32 model weights.
 *
 * Checkpoints embed weights as base64 so the existing Tiny GPT loaders can read
 * a browser-trained model without a new file format. Encoding is chunked to
 * avoid building one huge binary string with `String.fromCharCode(...bytes)`.
 */

export function encodeFloat32(values: Float32Array): string {
  const bytes = new Uint8Array(
    values.buffer,
    values.byteOffset,
    values.byteLength,
  );
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(
      ...bytes.subarray(offset, offset + chunkSize),
    );
  }
  return btoa(binary);
}

export function decodeFloat32(value: string): Float32Array {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index++) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Float32Array(bytes.buffer);
}
