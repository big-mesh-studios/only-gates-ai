import intMathSource from "../../../../src/int_math.h?raw";
import tinyGptHeaderSource from "../../../../src/tiny_gpt.h?raw";
import quantizeSource from "../../../../src/tiny_gpt_quantize.c?raw";
import inferSource from "../../../../src/tiny_gpt_infer.c?raw";

/*
 * The deep-dive page shows the production source. Full files are imported
 * verbatim; chapter listings are extracted by marker so the page cannot drift
 * from the implementation. A missing marker is a build error.
 */

function slice(
  source: string,
  startMarker: string,
  label: string,
  endMarker = "\n}\n",
): string {
  // Markers assume LF. Normalize so the extractor also works on a Windows
  // checkout with autocrlf enabled.
  source = source.replace(/\r\n/g, "\n");
  const start = source.indexOf(startMarker);
  if (start < 0) {
    throw new Error(`deep-dive source extract: missing start marker for ${label}`);
  }
  const end = source.indexOf(endMarker, start);
  if (end < 0) {
    throw new Error(`deep-dive source extract: missing end marker for ${label}`);
  }
  return source.slice(start, end + endMarker.length);
}

export const sources = {
  intMath: intMathSource,
  header: tinyGptHeaderSource,
  quantize: quantizeSource,
  infer: inferSource,
} as const;

export const snippets = {
  q16FromFloat: slice(quantizeSource, "static int32_t q16_from_float(", "q16_from_float"),
  scaleQ16FromFloat: slice(
    quantizeSource,
    "static int32_t scale_q16_from_float(",
    "scale_q16_from_float",
  ),
  quantizeValue: slice(quantizeSource, "static int8_t quantize_value(", "quantize_value"),
  quantizeRows: slice(quantizeSource, "static void quantize_rows(", "quantize_rows"),
  quantizeColumns: slice(
    quantizeSource,
    "static void quantize_columns_transposed(",
    "quantize_columns_transposed",
  ),
  quantizeGain: slice(quantizeSource, "static void quantize_gain(", "quantize_gain"),
  buildGeluLut: slice(quantizeSource, "static void build_gelu_lut(", "build_gelu_lut"),
  inverseSqrtQ16: slice(quantizeSource, "static int32_t inverse_sqrt_q16(", "inverse_sqrt_q16"),
  allocateLayerWeights: slice(
    quantizeSource,
    "static int allocate_layer_weights(",
    "allocate_layer_weights",
  ),
  create: slice(quantizeSource, "TinyGpt *tiny_gpt_create(", "tiny_gpt_create"),
  quantizeActivationRow: slice(
    inferSource,
    "static void quantize_activation_row(",
    "quantize_activation_row",
  ),
  matmul: slice(inferSource, "static void matmul_q16(", "matmul_q16"),
  rmsnorm: slice(inferSource, "static void rmsnorm_q16(", "rmsnorm_q16"),
  gelu: slice(inferSource, "static int32_t gelu_q16(", "gelu_q16"),
  attention: slice(inferSource, "static void attention_q16(", "attention_q16"),
  softmax: slice(inferSource, "static void softmax_at_q15(", "softmax_at_q15"),
  forward: slice(inferSource, "static void forward_range_impl(", "forward_range_impl"),
  nextRandom: slice(inferSource, "static uint32_t next_random(", "next_random"),
  probabilities: slice(
    inferSource,
    "void tiny_gpt_probabilities(",
    "tiny_gpt_probabilities",
  ),
  sample: slice(inferSource, "int32_t tiny_gpt_sample(", "tiny_gpt_sample"),
} as const;

/** Full-file line count for the appendix summaries. */
export function lineCount(source: string): number {
  return source.replace(/\n$/, "").split("\n").length;
}
