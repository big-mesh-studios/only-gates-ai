declare module "*only-gates-ai.js" {
  const createModule: (
    options?: Record<string, unknown>,
  ) => Promise<unknown>;
  export default createModule;
}

declare module "*only-gates-ai-simd.js" {
  const createModule: (
    options?: Record<string, unknown>,
  ) => Promise<unknown>;
  export default createModule;
}
