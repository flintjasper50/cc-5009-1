// Minimal ambient declarations so the unit-test runner typechecks without pulling in
// the full @types/node package, which is not a dependency of this project.

declare module 'node:assert/strict' {
  interface Assert {
    (value: unknown, message?: string): asserts value;
    ok(value: unknown, message?: string): asserts value;
    equal(actual: unknown, expected: unknown, message?: string): void;
    notEqual(actual: unknown, expected: unknown, message?: string): void;
    deepEqual(actual: unknown, expected: unknown, message?: string): void;
    throws(fn: () => unknown, message?: string): void;
  }
  const assert: Assert;
  export default assert;
}

declare const process: {
  exit(code?: number): never;
  argv: string[];
  env: Record<string, string | undefined>;
};
