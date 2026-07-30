# 05 RoPE：旋转位置编码 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

## 总任务

补全 `main.cpp` 中 `rope(q, k, pos)` 的四个 TODO：对**一个位置**的 q/k 切片原地旋转。

模块 04 算出的 q / k 投影只是"内容"向量，不含任何位置信息——把句子打乱，注意力分数完全不变。RoPE（Rotary Position Embedding）在注意力之前对 q 和 k 做一次**与位置相关的旋转**，使得两个 token 的 q·k 点积只取决于它们的**相对距离**，这正是注意力需要的位置信号。

```
embedding → RMSNorm → q/k/v 投影(模块 04) → 【RoPE(本模块)】 → k 写入 key cache → 注意力(模块 06)
```

**输入**：data.h 里的 const 数组，无需读任何文件。数据来自真实前向传播：stories15M 第 0 层，提示词 `"Once upon a time"` → token id `[1, 9038, 2501, 263, 931]`，共 P=5 个位置。

| 变量 | 位置 | 形状 | 布局 | 含义 | 来自模型哪里 |
| --- | --- | --- | --- | --- | --- |
| `kQ` | data.h | (P, dim) = (5, 288) | pos-major | RoPE 之前的 q 投影 | 第 0 层 `x @ wq`，即 5 个提示词位置的 query |
| `kK` | data.h | (P, kv_dim) = (5, 288) | pos-major | RoPE 之前的 k 投影 | 第 0 层 `x @ wk`，即 5 个提示词位置的 key |

模型常量：`dim = 288`、`kv_dim = 288`、`n_heads = n_kv_heads = 6`、`head_size = 48`、`P = 5`（代码中为 `kDim`、`kKvDim`、`kHeadSize`、`kPositions`）。data.h 由 `../tools/embed_data.py` 从 `data/*.txt` 生成，值与 txt 完全一致，请勿手改。

**输出**：`main()` 已经写好——从 data.h 拷贝输入、对 pos = 0..4 逐位置调用 `rope`、写出 `out_q.txt` / `out_k.txt`。`data/expected_*` 是黄金数据，不要修改。

## 子任务一：以步长 2 遍历相邻维度对

以步长 2 遍历相邻维度对 `i = 0, 2, 4, ... dim-2`。

需要的知识——内存布局。两个数组都是 **pos-major 拼接**——位置 pos 占据 `[pos*dim, (pos+1)*dim)`；每个位置内部是 **head-major**——head h 占据 `[h*head_size, (h+1)*head_size)`（head_size = 48，6 个 head 共 288 维）。因此下标 i 所属 head 为 `i / head_size`，head 内维度对索引为 `i % head_size`。

## 子任务二：计算 `head_dim`、`freq` 与 `angle`

对每个维度对，计算 `head_dim = i % head_size`、`freq = 1 / 10000^(head_dim / head_size)`、`angle = pos * freq`，并用 float 版本的 `std::cos` / `std::sin`。

需要的知识——数学原理。把每个 head 内的相邻两维 `(v0, v1)` 视作一个二维向量（等价于一个复数），按角度 `pos * freq` 旋转：

```
for i = 0, 2, 4, ... dim-2:
    head_dim = i % head_size                     # 该维度对在所属 head 内的索引
    freq     = 1 / 10000^(head_dim / head_size)  # 每个维度对有自己的频率
    angle    = pos * freq                        # 旋转角随位置线性增长
    (v0', v1') = (v0*cos(angle) - v1*sin(angle),  v0*sin(angle) + v1*cos(angle))
```

**频率按 head 内索引计算**，不是全局索引：每个 head 的前 24 个维度对（head_size=48）各用一套从 1 到 ~1e-4 的频率，低频转得慢（编码长距离），高频转得快（编码局部）。

## 子任务三：原地旋转 q 中的该维度对

原地旋转 q 中的该维度对：`(v0', v1') = (v0*cos - v1*sin, v0*sin + v1*cos)`。

需要的知识——**原地旋转**：旋转后的 k 就地写回，随后原样存入 key cache——即 **cache 里保存的是旋转后的 k**。

## 子任务四：在 `i < kv_dim` 约束下旋转 k 中的同一维度对

仅当 `i < kv_dim` 时，对 k 中的同一维度对做同样旋转。

需要的知识——**q 全部 dim 个值都旋转，k 只旋转前 kv_dim 个值**（代码中第二个旋转包在 `if (i < kv_dim)` 里）。原因：q 有 n_heads × head_size = dim 个值，而 k 只有 n_kv_heads × head_size = kv_dim 个值；在 GQA 模型中 n_kv_heads < n_heads，kv_dim < dim，越过 kv_dim 就越界了。本模型 kv_dim == dim == 288，所以 k 实际上也全部旋转，但边界判断这个约定必须保留。

## 构建 / 运行 / 验证

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out_q.txt data/expected_q.txt
python3 ../tools/compare.py out_k.txt data/expected_k.txt
```

## 常见错误

- 所有频率都对不上 → 忘记 `head_dim = i % head_size`，直接用了全局下标 i。
- 有系统性微小偏差 → `std::cos` / `std::sin` 走了 double 重载再转回 float（角度、频率、三角函数全程用 float）。
- pos=0 是免费自检：此时所有 angle = 0，q/k 必须原样通过；如果你改动了 pos=0 的数据，说明越界写或拷贝错了。
- 把 k 的循环上界写成 dim 而无 `i < kv_dim` 判断 → 本模块能过（kv_dim == dim），但换成 GQA 模型就会越界。

## 完成标准

两项比较全部 PASS。与模块 06 的衔接：本模块输出的 `out_k.txt`（旋转后的 k）就是 key cache 中保存的内容，`out_q.txt` 则直接用于与 cache 里的 k 做点积。`solution.cpp` 是参考答案——卡住再看，看完关掉自己写。
