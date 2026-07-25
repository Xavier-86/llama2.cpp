# 03 rmsnorm / softmax：两个小型内核 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

> 目标：实现 Transformer 中随处可见的两个归一化函数，每个函数都不到 10 行。

## RMSNorm

对于长度为 n 的向量 x 和学习得到的权重 w：

```
ss = (1/n) * sum(x_i^2)      # 均方
ss = 1 / sqrt(ss + 1e-5)     # eps 防止除零
out_i = w_i * ss * x_i
```

它归一化向量幅度以保证数值稳定。用于注意力之前、FFN 之前，以及模型末尾。

## Softmax（数值稳定版）

```
m = max(x)
out_i = exp(x_i - m) / sum_j exp(x_j - m)
```

减去最大值可防止 `exp(1000)` 溢出为 inf；数学结果不变，但数值计算中不可或缺。

## 数据文件

| 文件 | 内容 |
| --- | --- |
| `input_rmsnorm_x.txt` / `input_rmsnorm_w.txt` | 8 维示例：输入与权重 |
| `expected_rmsnorm.txt` | 预期输出 |
| `input_rmsnorm_x_real.txt` | 真实的 288 维输入（提示词首 token 的 embedding 行） |
| `expected_rmsnorm_real.txt` | 使用第 0 层 rms_att_weight 归一化的结果 |
| `input_softmax.txt` / `expected_softmax.txt` | 普通 8 维示例 |
| `input_softmax_big.txt` / `expected_softmax_big.txt` | 约为 1000 的值，用于检验稳定性技巧 |

## 任务与验证

真实尺寸示例需使用模块 01 的加载器，从 checkpoint 获取第 0 层 `rms_att_weight` 切片（偏移 0、长度 dim）。

```bash
python3 ../tools/compare.py out.txt data/expected_rmsnorm.txt
python3 ../tools/compare.py out_real.txt data/expected_rmsnorm_real.txt
python3 ../tools/compare.py out_sm.txt data/expected_softmax.txt
python3 ../tools/compare.py out_big.txt data/expected_softmax_big.txt
```

若 `big` 示例全部输出 0 或 NaN，说明忘记减去最大值。

## 提示

- 全部使用 `float` 计算（包括累加和），以保持与标准数据一致。
- Softmax **原地修改**输入；注意力会针对共享缓冲区中每个 head 的切片调用它。
