#ifndef TESTS_TEST_COMMON_H
#define TESTS_TEST_COMMON_H

#include <math.h>
#include <stdio.h>

static int test_failures = 0;

static void expect_near(const char *name, float actual, float expected,
                        float tolerance) {
  if (!isfinite(actual) || fabsf(actual - expected) > tolerance) {
    printf("FAIL %s: got %.8f, expected %.8f\n", name, actual, expected);
    ++test_failures;
  }
}

static void expect_true(const char *name, int condition) {
  if (!condition) {
    printf("FAIL %s\n", name);
    ++test_failures;
  }
}

static int test_report(void) {
  if (test_failures != 0) {
    printf("%d test(s) failed\n", test_failures);
    return 1;
  }
  printf("all tests passed\n");
  return 0;
}

#endif
