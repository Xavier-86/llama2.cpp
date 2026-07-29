# 06 attention：多头因果注意力 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

> 目标：实现 Transformer 的核心——带 KV cache 的多头因果注意力。这是最容易出现隐蔽错误的模块，请预留更多时间。

## 背景

每一层的前向传播中，注意力让当前位置"回看"它之前的所有位置：用 query 和历史上每个位置的 key 算相似度，归一化成权重后对 value 做加权和。生成是自回归的——位置 pos 只能看到 0..pos（**因果掩码**），所以推理时把每步算好的 K/V 存进 **KV cache**，后续位置直接复用，不必重算整个前缀。

本模块给出第 0 层、提示词 "Once upon a time"（token id `[1, 9038, 2501, 263, 931]`，P=5）前 5 个位置的真实数据：q 和 k 已经过 RoPE 旋转（模块 05 的产物），v 不做旋转。本模块只做注意力本身；之后的 wo 投影与残差连接由模块 08 完成。

## 数学原理

把 dim=288 的 q 切成 `n_heads` 个 head，每个长 `head_size`。对单个位置 pos、单个 head h：

```
qh      = q[h*head_size : (h+1)*head_size]
score_t = (qh · k_t) / sqrt(head_size)     # t = 0..pos；只遍历 0..pos 就是因果掩码（t > pos 不可见）
att     = softmax(score_0..pos)            # 只对这 pos+1 个 score 做 softmax
out_h   = sum_t att_t * v_t                # 对 value 行加权和
```

最后把 6 个 head 的 `out_h` 拼回 288 维向量，即该层的注意力输出（wo 之前）。

- 除以 `sqrt(head_size)` 防止点积随维度增大而过强，避免 softmax 饱和。
- **GQA**：query head h 对应的 KV head 是 `kv_h = h / kv_mul`，`kv_mul = n_heads / n_kv_heads`。stories15M 中 n_heads = n_kv_heads = 6，故 kv_mul = 1、kv_h = h（每个 head 各有自己的 K/V）；GQA 模型（n_kv_heads < n_heads）下多个 query head 会共享同一组 K/V。请按通用 kv_mul 写代码，不要硬编码 kv_h = h。

### KV cache 的布局

真实推理中 cache 形状为 (n_layers, seq_len, kv_dim)：每个位置生成时，把该层新算出的 k、v 写入第 pos 行；之后所有位置都能读到 0..pos 行。本模块直接给你第 0 层 cache 的前 P=5 行切片，pos-major 摊平成一维：

```
kKCache[t * kv_dim + kv_h * head_size + i]   # 位置 t、KV head kv_h、第 i 维
```

即逻辑上的 `kKCache[pos][head][head_size]` 三层索引；v cache 同理。

## 输入数据

所有输入都是 data.h 里的 const 数组（由 `../tools/embed_data.py` 从 `data/*.txt` 生成，值完全一致，勿手改 data.h），无需读任何文件：

| 变量 | 位置 | 形状 | 布局 | 含义 | 来自模型哪里 |
| --- | --- | --- | --- | --- | --- |
| `kQ` | data.h | (5, 288) | pos-major：位置 pos 占 `[pos*288, (pos+1)*288)`；位置内部按 head 切分，head h 占 `h*48` 起的 48 个值 | 5 个位置的 query，已做 RoPE | 第 0 层：RMSNorm 后的 xb × `wq`，再经 RoPE（模块 05 的输出） |
| `kKCache` | data.h | (5, 288) | 同上，等价于 `cache[pos][kv_head][head_size]` | 第 0 层 key cache 的前 5 行，已做 RoPE | 第 0 层：xb × `wk` 经 RoPE 后写入 key cache |
| `kVCache` | data.h | (5, 288) | 同上 | 第 0 层 value cache 的前 5 行（V 不做 RoPE） | 第 0 层：xb × `wv` 写入 value cache |

模型常量（stories15M）：dim=288，hidden_dim=768，n_layers=6，n_heads=6，n_kv_heads=6，vocab_size=32000，seq_len=256，head_size=48，kv_dim=288。代码中对应 `kDim` / `kNumHeads` / `kHeadSize` / `kKvMul`（=1）/ `kKvDim` / `kPositions`（=5）。

## 任务

补全 `main.cpp` 中的 5 个 `TODO`：

1. **task 1**：遍历 head h = 0..5，求 `kv_h = h / kv_mul`，从 q 中切出本 head 的 `qh`（`h * head_size` 起的 48 个值）。
2. **task 2**：对 t = 0..pos（仅历史+当前，这就是因果掩码），从 k_cache 第 t 行切出 kv_h 的 key（起点 `t * kv_dim + kv_h * head_size`），算点积 `qh · key` 并除以 `sqrt(head_size)`，存入本 head 的 att 切片下标 t。
3. **task 3**：只对本 head att 切片的 0..pos 段做 softmax——复用给定的 `softmax`，它在共享缓冲区上**原地**工作（模块 03 的约定），未来位置的槽位不能参与 exp。
4. **task 4**：把本 head 的 out 切片清零，按与 K 相同的行/head 切法从 v_cache 取行，累加 `att_t * v_t`。
5. **task 5**（在 `main()` 里）：循环结束后 att 里留的是最后位置 pos=4 的权重；按 head-major 收集（head 0 的 5 个、head 1 的……共 30 个）写入 out_att.txt。

`main()` 的其余部分已写好：对 pos = 0..4 逐位置调用 `attention`，输出 out.txt（5×288，所有位置拼接）。

## 构建 / 运行 / 验证

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out.txt     data/expected_out.txt
python3 ../tools/compare.py out_att.txt data/expected_att_weights_lastpos.txt
```

## 常见错误

- **先对 att_weights 再对完整输出**：权重对而输出错 → 问题在加权求和或 head 切片；权重错 → 查点积或 softmax 范围。
- 对整个 scratch（或 seq_len=256）而非 0..pos 段做 softmax → 未来位置的 0 也被 exp 进去。
- 忘记除以 `sqrt(head_size)`。
- 行内偏移错：第 t 行、第 kv_h 个 head 的起点是 `t * kv_dim + kv_h * head_size`。
- 没有按通用 `kv_mul` 写（硬编码 kv_h = h）→ 本模块能过，但换成 GQA 模型就错。

## 完成标准

两项比较全部 PASS。`solution.cpp` 是参考答案——卡住再看，看完关掉自己写。
