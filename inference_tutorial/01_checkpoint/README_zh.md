# 01 checkpoint：加载模型权重 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

## 总任务

补全 `main.cpp` 中的两个 TODO：解析 `../../stories15M.bin` 的头部，并映射其权重区中的 11 个权重张量：

```
[Config 头部：7 x int32] [权重区：紧密排列的 float32]
        |                           |
        v                           v
 out_config.txt              out_summary.txt
   （7 个整数）             （11 个张量 x 3 个值）
```

checkpoint 文件就是一个小二进制头加上按固定顺序紧密排列的全部 float32 权重：先解析头部拿到各维度，再按顺序遍历权重区，为每个张量记录一个视图。

这是全教程唯一专门讲二进制解析的模块。之后的模块不再手写解析代码：需要权重时直接调用 `../common/checkpoint.h` 里的 `tut::load_checkpoint()`——它就是你本模块答案的复用封装。所以这里值得把格式彻底弄懂。

**输入**：只有一个二进制文件，且 `main()` 已写好的部分已经把它读入内存——文件流的读取不需要你来做：

| 变量 | 位置 | 形状 | 含义 |
| --- | --- | --- | --- |
| `checkpoint_path` | main.cpp | — | 指向 `../../stories15M.bin`（stories15M，FP32） |
| `buf` | main.cpp，已填好 | (file_bytes/4,) float | 整个文件：Config 头部 + 权重区 |
| `Config` | main.cpp | 7 x int32 | 头部结构体：`dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len` |
| `Weights` | main.cpp | 12 个 span | 每个张量一个 `std::span<const float>` 视图，待你填充 |

**输出**：两个 TODO 要写两个文件——`out_config.txt`（7 个 Config 整数，每行一个）和 `out_summary.txt`（33 行：每个张量的元素数量、首元素、全部元素之和）。`data/expected_config.txt` 和 `data/expected_weight_summary.txt` 是黄金数据，不要修改；后者也给出了本模型各张量的实际大小。

## 子任务一：解析 Config 头部（TODO task 1）

从 `buf` 的起始处读出 7 个 int32 字段填入 `Config`，按存储原样（每行一个）输出到 `out_config.txt`，并用 `--exact` 与 `data/expected_config.txt` 比较验证。

需要的知识——Config 头部的布局与约定：

- 字段依次为：`dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len`
- `vocab_size` 为**负数**表示分类器权重不共享；stories15M 共享权重，因此读到 +32000

需要的知识——文件如何进入内存。`main()` 已经用 `ifstream` 把整个文件读入 `vector<float>`；15M 规模下这样做完全可行。进阶做法是 `mmap`（见 `../../run.cpp` 中的 `MappedFile`：零拷贝，对大模型至关重要）——两种方式在这里都行。

## 子任务二：遍历权重区并汇总各张量（TODO task 2）

权重区紧接在头部之后。按下方的布局表计算每个张量的偏移，按文件顺序为 `Weights` 的每个字段记录一个 `std::span` 视图。注意：

- `kv_dim = n_kv_heads * (dim / n_heads)`
- 旧版的 `freq_cis_real` / `freq_cis_imag` 表位于 `rms_final_weight` 和 `wcls` 之间——跳过它们
- 权重共享时 `wcls` 与 `token_embedding_table` 指向同一内存；仅在不共享时才单独读取

然后按表中顺序对 11 个张量各输出 3 行到 `out_summary.txt`（`std::scientific`，`std::setprecision(3)`）：**元素数量、首元素、全部元素之和（用 double 按文件顺序累加）**，共 33 个数，并与 `data/expected_weight_summary.txt` 比较验证。

需要的知识——权重区的布局（本模型的实际大小见 `data/expected_weight_summary.txt`）：

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

需要的知识——"映射"不需要复制，只需记录每个张量的 `(pointer, length)`；C++20 的 `std::span` 正好表达这种"视图"。第 l 层 wq 的切片起点为 `wq_start + l * dim * dim`，长度为 `dim * dim`，后续模块会频繁使用这种索引。

需要的知识——求和必须用 `double` 按文件顺序累加，否则无法与黄金数据一致。

## 构建 / 运行 / 验证

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out_config.txt data/expected_config.txt --exact
python3 ../tools/compare.py out_summary.txt data/expected_weight_summary.txt
```

## 常见错误

- 求和错误通常源于：张量顺序错误、大小错误，或忘记跳过 freq_cis。
- 大小都对但和与黄金数据不一致 → 没有用 `double` 累加，或没有按文件顺序累加。

## 完成标准

两项比较全部 PASS。`solution.cpp` 是参考答案——卡住再看，看完关掉自己写。
