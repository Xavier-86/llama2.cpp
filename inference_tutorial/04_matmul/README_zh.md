# 04 matmul：矩阵-向量乘法 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md)

> 目标：实现 `W (d,n) @ x (n,) -> xout (d,)`。模型 99% 的时间都花在此函数中。

## 定义

W 按行优先存储，共 d 行，每行 n 个 float：

```
xout[i] = sum_j  W[i*n + j] * x[j]        i = 0..d-1
```

W 的每一行与 x 做点积。在模型中，x 是激活向量（dim），W 是权重矩阵（如 wq），输出则是投影结果。

## 数据文件

| 文件 | 内容 |
| --- | --- |
| `input_w.txt` | 3x4 行优先矩阵，共 12 个数 |
| `input_x.txt` | 4 维向量 |
| `expected_out.txt` | 3 维结果 |

## 任务与验证

```bash
python3 ../tools/compare.py out.txt data/expected_out.txt
```

手算检查：第 0 行 = `0.5*0.5 + (-1.25)*(-1.0) + 2.0*2.0 + 0.75*0.25`。

## 提示

- 使用 `float` 累加（与参考实现一致；double 更精确，但会在 1e-6 量级偏离标准数据，在本容差下无害）。
- 正确性只是入场券，重点是**性能**：
  - 每个权重元素只参与一次乘加，因此工作量与读取字节数成正比，是典型的内存带宽受限任务。
  - 比较 `-O2` 与 `-O0`；尝试用 OpenMP/pthreads 并行外层循环，观察在 15M 规模下是否有帮助。
  - 模块 11 的 int8 版本将读取字节数降至 1/4，这就是 decode 加速的全部秘密。
