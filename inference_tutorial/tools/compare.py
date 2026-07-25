#!/usr/bin/env python3
"""Compare two data files (whitespace-separated numbers, one per line).

Golden data in this project is printed with 3 decimal places (setprecision(3)),
so the default tolerance is atol=1e-3 / rtol=1e-3 to match.

usage:
    python3 compare.py <your_output.txt> <expected.txt> [options]

options:
    --atol F   absolute tolerance (default 1e-3)
    --rtol F   relative tolerance (default 1e-3): |a-b| <= atol + rtol*|b| passes
    --exact    exact-equality mode (for token ids, int8 values, ...)
    --text     text mode: byte-for-byte comparison (for decode / generated text)
    -n N       print at most N mismatching positions (default 5)

exit code: 0 = PASS, 1 = FAIL (so comparisons can be chained in scripts).
"""

import argparse
import sys


def load_numbers(path):
    with open(path) as f:
        return [float(tok) for tok in f.read().split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actual")
    ap.add_argument("expected")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--text", action="store_true")
    ap.add_argument("-n", type=int, default=5)
    args = ap.parse_args()

    if args.text:
        with open(args.actual) as f:
            a = f.read()
        with open(args.expected) as f:
            b = f.read()
        if a == b:
            print(f"PASS  ({args.actual} == {args.expected}, {len(b)} bytes)")
            return 0
        print(f"FAIL  text differs: actual {len(a)} bytes, expected {len(b)} bytes")
        for i, (ca, cb) in enumerate(zip(a, b)):
            if ca != cb:
                print(f"  first difference at offset {i}: actual {ca!r} expected {cb!r}")
                print(f"  actual context:   {a[max(0, i - 20):i + 20]!r}")
                print(f"  expected context: {b[max(0, i - 20):i + 20]!r}")
                break
        return 1

    a = load_numbers(args.actual)
    b = load_numbers(args.expected)

    if len(a) != len(b):
        print(f"FAIL  length mismatch: actual {len(a)} values, expected {len(b)} values")
        return 1

    max_abs = 0.0
    max_rel = 0.0
    bad = []
    for i, (x, y) in enumerate(zip(a, b)):
        d = abs(x - y)
        max_abs = max(max_abs, d)
        if abs(y) > 1e-30:
            max_rel = max(max_rel, d / abs(y))
        if args.exact:
            ok = (x == y)
        else:
            ok = d <= args.atol + args.rtol * abs(y)
        if not ok:
            bad.append((i, x, y, d))

    mode = "exact" if args.exact else f"atol={args.atol:g} rtol={args.rtol:g}"
    if not bad:
        print(f"PASS  n={len(a)}  max|diff|={max_abs:.3e}  max|rel|={max_rel:.3e}  ({mode})")
        return 0
    print(f"FAIL  {len(bad)}/{len(a)} values out of tolerance ({mode})")
    print(f"      max|diff|={max_abs:.3e}  max|rel|={max_rel:.3e}")
    for i, x, y, d in bad[: args.n]:
        print(f"      [{i}] actual {x:.9g}  expected {y:.9g}  |diff|={d:.3e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
