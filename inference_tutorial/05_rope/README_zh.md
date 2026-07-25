# 05 RoPE：旋转位置编码 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md)

> 目标：向 Q 和 K 注入位置信息。注意力本身无法判断词序，正是这一步让它感知位置。

## 算法

将 Q（以及 K）中的每对相邻值 `(v0, v1)` 视作二维向量，按 `pos * freq` 旋转：

```
for i = 0, 2, 4, ... dim-2:
    head_dim = i % head_size                     # 在当前 head 内的维度对索引
    freq     = 1 / 10000^(head_dim / head_size)  # 每个维度对使用不同频率
    angle    = pos * freq
    (v0', v1') = (v0*cos(angle) - v1*sin(angle),  v0*sin(angle) + v1*cos(angle))
```

规则：

- 旋转 Q 的全部 `dim` 个值。
- 只旋转 K 的前 `kv_dim` 个值（此处 kv_dim = dim = 288，因此 K 也会全部旋转；仅在 kv_dim < dim 的 GQA 模型中才会体现差异）。
- **原地**旋转：旋转后的 K 保留在 KV cache 中，即 cache 保存的是旋转后的 K。
- 不同频率编码不同距离尺度：低频旋转慢（长距离），高频旋转快（局部）。

最终效果是：两个 token 的 Q·K 点积只取决于它们的**相对距离**，这正是注意力需要的位置信号。

## 数据文件（位置优先，P=5 个位置 x 288 个值）

| 文件 | 内容 |
| --- | --- |
| `input_q.txt` | 投影后、旋转前的 Q（5 x 288） |
| `input_k.txt` | 旋转前的 K（5 x 288） |
| `expected_q.txt` | 旋转后的 Q |
| `expected_k.txt` | 旋转后的 K（即 KV cache 保存的内容） |

数据来自真实前向传播：第 0 层，提示词 `"Once upon a time"`。

## 任务与验证

加载 Q/K，对 pos = 0..4 的每个位置进行旋转并比较：

```bash
python3 ../tools/compare.py out_q.txt data/expected_q.txt
python3 ../tools/compare.py out_k.txt data/expected_k.txt
```

## 提示

- 最常见错误是忘记 `head_dim = i % head_size` 而直接使用 i，导致所有频率错误。
- pos=0 时所有角度均为 0，Q/K 必须保持不变，可用作免费自检。
- 使用 `std::cos` / `std::sin` 的 float 重载，不要先算 double 再转换。
