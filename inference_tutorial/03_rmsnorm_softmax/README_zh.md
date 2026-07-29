# 03 rmsnorm / softmax：两个小型内核 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

> 目标：实现 Transformer 中随处可见的两个归一化函数，每个函数都不到 10 行。

## 背景

一次前向传播中，这两个函数被反复调用：

```
token embedding → [ RMSNorm → attention(内含 softmax) → RMSNorm → FFN ] × 6 层 → RMSNorm → logits → softmax(采样时)
```

RMSNorm 把向量幅度归一化以保证数值稳定；softmax 把任意实数向量变成概率分布（注意力权重、采样概率）。

## 数学原理

### RMSNorm

对于长度为 n 的向量 x 和学习得到的权重 w：

```
ss = (1/n) * sum(x_i^2)      # 均方
ss = 1 / sqrt(ss + 1e-5)     # eps 防止除零
out_i = w_i * ss * x_i
```

### Softmax（数值稳定版）

```
m = max(x)
out_i = exp(x_i - m) / sum_j exp(x_j - m)
```

减去最大值不改变数学结果，但能防止 `exp(1000)` 溢出为 inf——这是数值计算中不可或缺的技巧。

## 输入数据

所有输入都是代码里的 const 变量，无需读任何文件：

| 变量 | 位置 | 形状 | 含义 |
| --- | --- | --- | --- |
| `kRmsXToy` / `kRmsWToy` | main.cpp | (8,) | 教学用小例子：输入向量与权重 |
| `kRmsNormXReal` | data.h | (288,) | 真实输入：提示词首 token（`<s>`，id=1）的 embedding 行 |
| `kRmsNormWReal` | data.h | (288,) | 第 0 层 `rms_att_weight` 切片（checkpoint 权重区中 `rms_att_weight` 张量的前 dim 个元素） |
| `kSoftmaxIn` | main.cpp | (8,) | 普通 softmax 输入 |
| `kSoftmaxBig` | main.cpp | (4,) | 约等于 1000 的值，检验稳定性技巧 |

模型常量：`dim = 288`（代码中为 `kDim`）。data.h 由 `../tools/embed_data.py` 从 `data/*.txt` 生成，值与 txt 完全一致。

## 任务

补全 `main.cpp` 中的两个函数（各有 `TODO` 标注）：

1. **task 1 — `rmsnorm(o, x, weight)`**：按上面公式计算，结果写入 `o`。累加和也用 `float`，否则与标准数据对不上。
2. **task 2 — `softmax(x)`**：**原地**修改 `x`（注意力会对共享缓冲区中每个 head 的切片调用它）；先减最大值再 exp。

`main()` 已经写好：它用 4 组输入调用你的内核并写出 4 个输出文件。

## 构建 / 运行 / 验证

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out.txt      data/expected_rmsnorm.txt
python3 ../tools/compare.py out_real.txt data/expected_rmsnorm_real.txt
python3 ../tools/compare.py out_sm.txt   data/expected_softmax.txt
python3 ../tools/compare.py out_big.txt  data/expected_softmax_big.txt
```

## 常见错误

- `out_big.txt` 全为 0 或 NaN → 忘记在 exp 前减去最大值。
- `out_real.txt` 有系统性偏差 → 累加和用了 `double`，或 eps 放错了位置（先加 eps 再取倒数开方）。
- softmax 没有原地修改 → 本模块测试能过，但模块 06 的注意力会用到这个约定。

## 完成标准

四项比较全部 PASS。`solution.cpp` 是参考答案——卡住再看，看完关掉自己写。
