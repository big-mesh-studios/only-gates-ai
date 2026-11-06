/** Error codes returned by the C API. */
export enum OnlyGatesErrorCode {
  None = 0,
  OutOfMemory = 1,
  InvalidShape = 2,
  InvalidParams = 3,
}

const messages: Record<number, string> = {
  [OnlyGatesErrorCode.OutOfMemory]: "The WebAssembly module ran out of memory.",
  [OnlyGatesErrorCode.InvalidShape]: "The circuit shape is invalid.",
  [OnlyGatesErrorCode.InvalidParams]: "The circuit parameters are invalid.",
};

export class OnlyGatesError extends Error {
  readonly code: OnlyGatesErrorCode;

  constructor(code: OnlyGatesErrorCode, message?: string) {
    super(message ?? messages[code] ?? `only-gates-ai error ${code}`);
    this.name = "OnlyGatesError";
    this.code = code;
  }
}
