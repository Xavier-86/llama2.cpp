# 07 FFN：SwiGLU 前馈网络 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

> 目标：实现每层的第二个大模块。注意力在**不同位置之间**混合信息；FFN **独立处理每个位置**，并包含模型的大部分参数。

## 算法

```
h1 = W1 @ x          # (hidden_dim, dim) @ (dim,) -> (hidden_dim,)，升维投影
h3 = W3 @ x          # 相同形状，第二个分支
h  = silu(h1) * h3   # 逐元素；silu(v) = v * sigmoid(v) = v / (1 + exp(-v))
out = W2 @ h         # (dim, hidden_dim) @ (hidden_dim,) -> (dim,)，降维投影
```

两个升维投影，其中一个经过 SiLU 激活后与另一个逐元素相乘，再进行降维投影；这种结构称为 **SwiGLU**。

模型参数：dim=288、hidden_dim=768。使用模块 01 的加载器获取第 0 层 W1/W2/W3 切片（层偏移为 0）。

## 数据文件（真实前向传播，第 0 层，最后位置 pos=4）

| 文件 | 内容 |
| --- | --- |
| `input_x.txt` | FFN 输入（rms_ffn 归一化后），288 个值 |
| `expected_hidden.txt` | SwiGLU 后的隐藏状态，768 个值 |
| `expected_out.txt` | W2 后的输出（残差相加之前），288 个值 |

## 任务与验证

```bash
python3 ../tools/compare.py out_h.txt data/expected_hidden.txt
python3 ../tools/compare.py out.txt data/expected_out.txt
```

## 提示

- 先检查 hidden，再检查 out：hidden 错误时检查 W1/W3 切片或 silu；hidden 正确而 out 错误则检查 W2。
- 使用 `exp` 的 float 版本。
- 不要颠倒矩阵方向：W1 为 (hidden_dim, dim)，即 hidden_dim 行、每行 dim 个元素。若 matmul 从缓冲区大小推导维度（`xout.size()` = d、`x.size()` = n），就不易出错。
