# 07 FFN：SwiGLU 前馈网络 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

## 总任务

补全 `main.cpp` 中的 4 个 TODO，实现每层 Transformer 块的第二个大模块——SwiGLU 前馈网络。注意力在**不同位置之间**混合信息；FFN **独立处理每个位置**，并包含模型的大部分参数。

每个 Transformer 层里，注意力之后紧跟一个 FFN：

```
x → RMSNorm(rms_ffn_weight) → FFN(SwiGLU) → 残差相加
```

本模块只做中间的 FFN 部分：输入 `kX` 是真实前向传播中、第 0 层最后一个 prompt 位置（pos=4，prompt 为 "Once upon a time"，共 P=5 个 token）经过 `rms_ffn_weight` 归一化后的向量；输出是残差相加之前的值。**SwiGLU** 由两个升维投影、一个门控激活和一个降维投影组成：

```
h1  = W1 @ x          # (hidden_dim, dim) @ (dim,) -> (hidden_dim,)，升维投影
h3  = W3 @ x          # 相同形状，第二个分支
h   = silu(h1) ⊙ h3   # 逐元素相乘；silu(v) = v * sigmoid(v) = v / (1 + e^(-v))
out = W2 @ h          # (dim, hidden_dim) @ (hidden_dim,) -> (dim,)，降维投影
```

其中一个升维投影经过 SiLU 激活后与另一个逐元素相乘（门控），再投影回原维度。

**输入**：只有两类——`data.h` 里的 const 数组 `kX`，以及从 checkpoint 切出来的权重。无需在本模块读任何文件。

| 变量 | 位置 | 形状 | 布局 | 含义 | 来自模型哪里 |
| --- | --- | --- | --- | --- | --- |
| `kX` | data.h | (288,) | 一维 float 数组 | FFN 输入：最后一个 prompt 位置经 `rms_ffn_weight` 归一化后的 x | 第 0 层 FFN 入口（`data/input_x.txt` 的镜像） |
| `ckpt.weights.w1` | `tut::load_checkpoint("../../stories15M.bin")` | (6, 768, 288) = (n_layers, hidden_dim, dim) | 行优先，每层 hidden_dim 行 × dim 列 | 升维投影权重（门控分支） | checkpoint 权重区 `w1` 张量，层 l 切片起点 = `l * hidden_dim * dim` |
| `ckpt.weights.w3` | 同上 | (6, 768, 288) | 行优先，同 w1 | 升维投影权重（线性分支） | checkpoint 权重区 `w3` 张量，层 l 切片起点 = `l * hidden_dim * dim` |
| `ckpt.weights.w2` | 同上 | (6, 288, 768) = (n_layers, dim, hidden_dim) | 行优先，每层 dim 行 × hidden_dim 列 | 降维投影权重 | checkpoint 权重区 `w2` 张量，层 l 切片起点 = `l * dim * hidden_dim` |

本模块只用第 0 层（l = 0，偏移为 0）。模型常量（stories15M）：`dim = 288`、`hidden_dim = 768`、`n_layers = 6`、`n_heads = 6`、`n_kv_heads = 6`、`vocab_size = 32000`、`seq_len = 256`、`head_size = 48`、`kv_dim = 288`。代码中 `dim`/`hidden` 直接从 checkpoint 头部读取，不写死。

`data.h` 由 `../tools/embed_data.py` 从 `data/*.txt` 生成，值与 txt 完全一致；**不要手改**，重新转储后用 `python3 ../tools/embed_data.py ..` 重新生成。

**输出**：`main()` 的骨架已经写好——它加载 checkpoint、准备缓冲区并负责写输出文件：`out_h.txt`（门控后的隐向量 `h`，768 个 float）和 `out.txt`（FFN 输出，288 个 float）。`data/expected_hidden.txt` 和 `data/expected_out.txt` 是黄金数据，不要修改。

## 子任务一：切权重

用 `subspan` 从 `ckpt.weights.w1/w2/w3` 中切出第 0 层的 W1/W2/W3，每层偏移见上面的输入表（这里 l = 0，偏移都是 0，但切片长度仍然要取对）。

需要的知识——checkpoint 的权重布局。checkpoint 的二进制解析由 `../common/checkpoint.h` 的 `tut::load_checkpoint()` 完成，返回的 `Weights` 里每个张量都是指向同一缓冲区的 `std::span` 视图，按层连续排布，用 `subspan` 切片即可。`w1`/`w3` 每层 `hidden_dim * dim` 个元素（层 l 切片起点 = `l * hidden_dim * dim`），`w2` 每层 `dim * hidden_dim` 个（层 l 切片起点 = `l * dim * hidden_dim`）。归一化权重 `rms_ffn_weight`（层 l 起点 = `l * dim`）本模块不直接使用——输入 `kX` 已经是归一化后的值。

## 子任务二：升维投影 `h1 = W1 @ x` 和 `h3 = W3 @ x`

用 `matmul()` 计算两个升维投影；`x` 就是 `data.h` 里的 `kX`，无需加载。

需要的知识——`matmul` 如何看待权重。`matmul` 在本模块已给定（模块 04 会从头实现它）：W (d,n) @ x (n,) -> xout (d,)，`w` 按行优先存 d 行、每行 n 个元素。它从缓冲区大小推导维度（`xout.size()` = 行数、`x.size()` = 列数），所以只要切片长度给对，W1/W3 的方向——(hidden_dim, dim)，即 hidden_dim 行、每行 dim 个元素——就不可能搞反。

## 子任务三：SwiGLU 门控 `h = silu(h1) ⊙ h3`

逐元素施加门控，结果留在 `h1` 里；`main()` 随后把它写出到 `out_h.txt`。

需要的知识——SiLU 激活函数：`silu(v) = v * sigmoid(v) = v / (1 + e^(-v))`。用 `exp` 的 float 版本，累加也用 float——用 `double` 计算会与黄金数据对不上。

## 子任务四：降维投影 `out = W2 @ h`

用 `matmul()` 把门控后的隐向量投影回原维度；结果就是 FFN 的输出——残差相加之前的值，`main()` 把它写出到 `out.txt`。

需要的知识——W2 的方向与 W1/W3 相反：它是 (dim, hidden_dim)，即 dim 行、每行 hidden_dim 个元素，所以调用是 `matmul(out, h1, w2)`，其中 `out` 大小为 dim，`h1` 大小为 hidden_dim。

## 构建 / 运行 / 验证

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out_h.txt data/expected_hidden.txt
python3 ../tools/compare.py out.txt data/expected_out.txt
```

## 常见错误

- 先查 `out_h.txt` 再查 `out.txt`：hidden 错了查 W1/W3 切片或 silu；hidden 对了而 out 错，问题在 W2 切片。
- 矩阵方向搞反：W1 是 (hidden_dim, dim)，即 hidden_dim 行、每行 dim 个元素。`matmul` 从缓冲区大小推导维度（`xout.size()` = 行数、`x.size()` = 列数），切片长度给对就不会反。
- 切片偏移算错：`w1`/`w3` 每层 `hidden_dim * dim` 个元素，`w2` 每层 `dim * hidden_dim` 个；`rms_ffn_weight` 是每层 `dim` 个，别混用。
- `silu` 用了 `double` 的 `exp` 或累加用了 `double` → 与黄金数据对不上；全程用 float。

## 完成标准

两项比较全部 PASS。`solution.cpp` 是参考答案——卡住再看，看完关掉自己写。
