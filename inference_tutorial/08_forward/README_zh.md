# 08 forward：组装完整的单步前向传播（FP32） <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

> 目标：将模块 01～07 组装为完整的 `forward(token, pos) -> logits`。这是 FP32 的总装，也是第一个里程碑验收。

## 完整流程（每一行都对应此前的一个模块）

```
x = embedding[token]                          # 查表（01）
for l in 0..n_layers-1:
    xb = rmsnorm(x, rms_att_weight[l])        # （03）
    q = Wq[l] @ xb;  k = Wk[l] @ xb;  v = Wv[l] @ xb   # （04）
    k_cache[l][pos] = k;  v_cache[l][pos] = v # 先写 cache
    rope(q, k_cache[l][pos], pos)             # 原地旋转（05）
    xb = attention(q, k_cache[l], v_cache[l], pos)     # （06）
    x += Wo[l] @ xb                           # 投影 + 残差
    xb = rmsnorm(x, rms_ffn_weight[l])
    x += ffn(xb, W1[l], W2[l], W3[l])         # （07）+ 残差
x = rmsnorm(x, rms_final_weight)
logits = Wcls @ x                             # 与 embedding table 共享
```

关键点：

- **pre-norm + residual**：归一化后的输出进入分支，原始 x 保留用于残差相加；每层两个 block、两次相加。
- KV cache 形状为 (n_layers, seq_len, kv_dim)：每层、每个位置各一行。
- 一次调用只处理一个 token；N 个 token 的提示词需要按顺序调用 N 次（即 prefill 阶段）。

## 数据文件

| 文件 | 内容 |
| --- | --- |
| `input_tokens.txt` | 提示词的 5 个 token id |
| `expected_logits.txt` | 每个位置的完整 logits，5 x 32000，位置优先 |
| `expected_argmax.txt` | 每个位置 logits 的 argmax，共 5 个整数 |

## 任务与验证

按顺序对每个 pos 调用 forward，并收集 logits：

```bash
python3 ../tools/compare.py out_argmax.txt data/expected_argmax.txt --exact
python3 ../tools/compare.py out_logits.txt data/expected_logits.txt
```

**先比较 argmax**：若一致，信息流基本正确。由于求和顺序差异，logit 约 1e-3 的差异正常；超过 1e-2 通常是真正的错误。

## 排错路线（logits 不一致时，从上到下检查）

1. 暂时只运行 1 层，可快速判断是否有层索引错误。
2. 回到模块 05/06/07 的标准数据，重新验证每个 block。
3. 检查层索引：第 l 层 wq 偏移为 `l * dim * dim`；第 l 层 cache 偏移为 `l * seq_len * kv_dim`。
4. 检查残差：应为 `x += branch_output`，而非 `x = branch_output`。

通过后，你就重建了 `run.cpp` 的 `Transformer::forward`。
