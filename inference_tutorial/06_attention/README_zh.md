# 06 attention：多头因果注意力 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md)

> 目标：实现 Transformer 的核心——带 KV cache 的多头注意力。这是最容易出现隐蔽错误的模块，请预留更多时间。

## 算法（单层、单个位置 pos）

输入：旋转后的 Q（dim,），以及 KV cache 中第 0..pos 行的 K 和 V（每行 kv_dim）。

将 dim 拆为 `n_heads` 个 head，每个长度为 `head_size`。对每个 head h：

```
qh = Q[h*head_size : (h+1)*head_size]
for t = 0..pos:                                # 仅历史和当前位置，即因果掩码
    kh_t = K_cache[t][kv_h*head_size : ...]    # kv_h = h / kv_mul（GQA 共享；此处 kv_mul=1，故 kv_h=h）
    score_t = (qh . kh_t) / sqrt(head_size)
att = softmax(score)                           # 只在 0..pos 区间计算
out_h = sum_t att_t * V_cache[t][kv_h slice]
```

将 6 个 head 的 `out_h` 拼回 dim 向量，即该层的注意力输出（之后还需 wo 投影和残差连接，由模块 08 完成）。

模型参数：n_heads=6、head_size=48、n_kv_heads=6、kv_mul=1、kv_dim=288。

## 数据文件（位置优先，P=5，每个位置 288 个值）

| 文件 | 内容 |
| --- | --- |
| `input_q.txt` | 每个位置旋转后的 Q |
| `input_k_cache.txt` | 每个位置的 K（旋转后，存于 cache） |
| `input_v_cache.txt` | 每个位置的 V |
| `expected_out.txt` | 每个位置的注意力输出（拼接各 head，wo 之前） |
| `expected_att_weights_lastpos.txt` | 最后位置 pos=4 的 softmax 权重：6 heads x 5 = 30 个值，head 优先 |

## 任务与验证

对 pos = 0..4，使用对应 Q 和 cache 的 0..pos 行计算输出：

```bash
python3 ../tools/compare.py out.txt data/expected_out.txt
python3 ../tools/compare.py out_att.txt data/expected_att_weights_lastpos.txt
```

## 提示

- **先比较 att_weights，再比较完整输出**：权重正确而输出错误，问题在加权求和或 head 切片；权重错误，则检查点积或 softmax 范围。
- 典型错误 1：对完整 seq_len（256）而非 0..pos 做 softmax，未来位置的 0 也会参与 exp。
- 典型错误 2：忘记除以 `sqrt(head_size)`。
- 典型错误 3：行内偏移错误。第 t 行、第 h 个 head 的起点为 `t * kv_dim + kv_h * head_size`。
- `t = 0..pos` 循环会随 pos 增长，因此注意力成本随序列长度线性增长。这正是 decode 必须缓存 K/V，而不能每步重算整个前缀的原因。
