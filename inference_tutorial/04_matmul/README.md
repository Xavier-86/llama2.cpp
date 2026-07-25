# 04 matmul: matrix-vector multiply <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md)

> Goal: implement `W (d,n) @ x (n,) -> xout (d,)`. The model spends 99% of its
> time inside this function.

## Definition

W is stored row-major: d rows of n floats each.

```
xout[i] = sum_j  W[i*n + j] * x[j]        i = 0..d-1
```

Each row of W gets dotted with x. In the model: x is an activation (dim), W is
a weight matrix (e.g. wq), and the output is a projection.

## Data files

| File | Content |
| --- | --- |
| `input_w.txt` | 3x4 matrix, row-major, 12 numbers |
| `input_x.txt` | 4-dim vector |
| `expected_out.txt` | 3-dim result |

## Tasks and verification

```bash
python3 ../tools/compare.py out.txt data/expected_out.txt
```

Sanity check by hand: row 0 = `0.5*0.5 + (-1.25)*(-1.0) + 2.0*2.0 + 0.75*0.25`.

## Hints

- Accumulate in `float` (matching the reference; double is more accurate but
  drifts from the golden data at the 1e-6 level — harmless at our tolerance).
- Correctness is only the entry ticket — **performance** is the point:
  - every weight element participates in exactly one multiply-add, so the work
    is proportional to bytes read: textbook memory-bandwidth-bound.
  - try `-O2` vs `-O0`; try multithreading the outer loop (OpenMP/pthreads) and
    see whether it helps at 15M scale.
  - module 11's int8 version cuts bytes read to 1/4 — that is the entire secret
    of faster decode.
