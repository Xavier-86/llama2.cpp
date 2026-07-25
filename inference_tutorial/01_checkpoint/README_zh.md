# 01 checkpoint：加载模型权重 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

> 目标：理解 `stories15M.bin` 文件格式，并映射其中的 11 个权重张量。后续所有模块都会从这里读取权重。

## 文件格式（llama2.c FP32 checkpoint）

```
[Config 头部：7 x int32] [权重区：紧密排列的 float32]
```

Config 字段依次为：`dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len`。注意：`vocab_size` 为**负数**表示分类器权重不共享；stories15M 共享权重，因此读到 +32000。

权重区按以下顺序排列（本模型的实际大小见 `data/expected_weight_summary.txt`）：

| # | 张量 | 形状 |
| --- | --- | --- |
| 1 | token_embedding_table | (vocab_size, dim) |
| 2 | rms_att_weight | (n_layers, dim) |
| 3 | wq | (n_layers, dim, dim) |
| 4 | wk | (n_layers, dim, kv_dim) |
| 5 | wv | (n_layers, dim, kv_dim) |
| 6 | wo | (n_layers, dim, dim) |
| 7 | rms_ffn_weight | (n_layers, dim) |
| 8 | w1 | (n_layers, hidden_dim, dim) |
| 9 | w2 | (n_layers, dim, hidden_dim) |
| 10 | w3 | (n_layers, hidden_dim, dim) |
| 11 | rms_final_weight | (dim,) |
| （跳过） | freq_cis_real + freq_cis_imag | 2 × (seq_len × head_size/2) 个 float：旧版预计算 RoPE 表，本项目不用，只需跳过 |
| 12 | wcls | 共享时与 token_embedding_table 指向同一内存；仅在不共享时读取 (vocab_size, dim) |

## 任务

1. 读取 Config，输出 7 个整数，与 `data/expected_config.txt` 做精确比较（`--exact`）。
2. 根据上表计算每个张量的偏移。每个张量输出 3 个值：**元素数量、首元素、全部元素之和（用 double 累加）**。共 33 个数，与 `data/expected_weight_summary.txt` 比较。

```bash
python3 ../tools/compare.py out_config.txt data/expected_config.txt --exact
python3 ../tools/compare.py out_summary.txt data/expected_weight_summary.txt
```

补全 `main.cpp`；`solution.cpp` 是参考答案。

## 提示

- 简单做法：用 `ifstream` 将整个文件读入 `vector<float>`。进阶做法：使用 `mmap`（见 `../../run.cpp` 中的 `MappedFile`），它零拷贝且对大模型至关重要。15M 规模下两种方式都可行。
- 按文件顺序用 `double` 累加求和，否则无法与标准数据一致。
- “映射”不需要复制，只需记录每个张量的 `(pointer, length)`；C++20 的 `std::span` 正好表达这种“视图”。
- 第 l 层 wq 的切片起点为 `wq_start + l * dim * dim`，长度为 `dim * dim`。后续模块会频繁使用这种索引。

## 完成标准

两项比较均 PASS。求和错误通常源于：张量顺序错误、大小错误，或忘记跳过 freq_cis。
