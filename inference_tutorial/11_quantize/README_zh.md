# 11 quantize：int8 量化推理 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md)

> 目标：将 FP32 实现升级为 int8 量化并重建 `runq.cpp`。改动集中在三个部分：权重存储格式、矩阵乘法，以及每次矩阵乘法前对激活进行量化。tokenizer、sampler 和生成循环均原样复用。

## 量化为什么更快

每个 decode 步骤都会读取一次全部权重；速度 ≈ 带宽 ÷ 字节数。int8 将权重读取量降至 1/4（每 32 个值额外带一个 float scale），因此吞吐量可提升近 4 倍——这就是本模块的全部动机。

## 11.1 量化 checkpoint 格式（`stories15M-q32.bin`）

**256 字节头部**：

```
u32 magic = 0x616b3432 ("ak42")
i32 version = 2
Config（7 x i32，与 FP32 相同）
u8  shared_classifier（1 表示 wcls 与 embedding 共享）
i32 group_size（GS，本文件中 = 32）
```

**权重区**（从偏移 256 开始，顺序如下）：

| # | 内容 |
| --- | --- |
| 1-3 | rms_att_weight、rms_ffn_weight（各 n_layers x dim）、rms_final_weight（dim）——仍为 **FP32** |
| 4 | q_tokens：int8（vocab x dim）+ float scales（vocab x dim / GS） |
| 5-7 | wq、wk、wv、wo：逐层存储，每个均为 int8（dim x dim）+ scales（dim x dim / GS） |
| 8-10 | w1、w2、w3：逐层存储，每个均为 int8（dim x hidden_dim）+ scales |
| 11 | shared=1 时不存在 wcls（复用 q_tokens） |

每个量化张量的布局：**先存全部 int8 值，再存全部 scale**；逐层张量排列为 `[L0.q][L0.s][L1.q][L1.s]...`。

加载时将 q_tokens 一次性反量化为 FP32 副本，用于 embedding 查表，从而避免每步重复反量化。

验证方法：输出 config + GS（8 个整数），用 `--exact` 与 `expected_config.txt` 比较；再输出类似模块 01 的权重摘要。`expected_weight_summary.txt` 先包含 3 个 FP32 张量各 3 行（size/first/sum），再包含 8 个量化张量各 5 行（q_total_size/q_first/q_sum/s_total_size/s_first）：

```bash
python3 ../tools/compare.py out_config.txt data/expected_config.txt --exact
python3 ../tools/compare.py out_summary.txt data/expected_weight_summary.txt
```

## 11.2 quantize / dequantize

对每组 GS 个值进行对称量化：

```
scale = max(|group values|) / 127
q[i]  = round(x[i] / scale)     # int8，范围 [-127, 127]
s[g]  = scale
dequantize: x[i] = q[i] * s[i / GS]
```

数据：`input_x.txt`（64 个值）-> `expected_q.txt`（64 个 int8，使用 `--exact`）、`expected_s.txt`（2 个 scale）、`expected_deq.txt`（往返结果）。GS=32。

注意：这里的 `round` 是中点值**远离零**舍入（`std::round`），不是银行家舍入。尤其是用 Python 交叉验证时要注意，Python 的 `round` 使用银行家舍入。

## 11.3 int8 矩阵乘法

```
对每一行：
    按 GS 大小分组：int8 x int8 -> int32 整数乘加
    （不会溢出：127*127*32 ~ 5*10^5）
    每组结束时：val += (float)group_int_sum * w.s[group] * x.s[group]
```

整数部分精确计算，只有组边界处出现浮点运算，因此量化版本既快又稳定。

数据（n=8、d=3、GS=4）：`input_matmul_wq.txt`（24 个 int8，行优先）、`input_matmul_ws.txt`（3x2 个 scale，行优先）、`input_matmul_xq.txt`（8 个 int8）、`input_matmul_xs.txt`（2 个 scale）-> `expected_matmul_out.txt`（3 个值）。

## 11.4 量化前向传播与生成

基于模块 08 的 forward，在**每次 matmul 之前**将激活量化为 int8，再通过 11.3 的 int8 权重执行乘法。RMSNorm、RoPE、attention、SwiGLU、softmax 均继续使用 FP32，不做修改。

数据：`input_tokens.txt`（相同的 5 个 token）-> `expected_argmax.txt`（5 个整数，使用 `--exact`）+ `expected_logits_lastpos.txt`（最后位置的完整 logits）。

最后运行 64 步贪心生成，与 `expected_greedy_ids.txt` / `expected_greedy_text.txt` 比较：

```bash
python3 ../tools/compare.py out_ids.txt data/expected_greedy_ids.txt --exact
python3 ../tools/compare.py out_text.txt data/expected_greedy_text.txt --text
```

## 任务与完整验证

`main.cpp` 保留此前模块已完成的 tokenizer、sampler、attention 和生成代码。补全四项量化专属任务：

1. 映射 256 字节 checkpoint 头部和 FP32/int8 混合权重区
2. 实现分组 quantize 与 dequantize
3. 实现分组 int8 矩阵-向量乘法
4. 将模块 08 的 FP32 投影替换为 quantize + int8 matmul

程序会输出分阶段排错所需的全部中间结果：

| 输出 | 标准数据 |
| --- | --- |
| `out_config.txt` | `expected_config.txt` |
| `out_summary.txt` | `expected_weight_summary.txt` |
| `out_q.txt`、`out_s.txt`、`out_deq.txt` | quantize/dequantize 输出 |
| `out_matmul.txt` | 独立 int8 matmul 输出 |
| `out_argmax.txt`、`out_logits.txt` | 5 个位置的量化前向传播 |
| `out_ids.txt`、`out_text.txt` | 64 步贪心生成 |

构建并运行：

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
```

逐阶段验证：

```bash
python3 ../tools/compare.py out_config.txt data/expected_config.txt --exact
python3 ../tools/compare.py out_summary.txt data/expected_weight_summary.txt
python3 ../tools/compare.py out_q.txt data/expected_q.txt --exact
python3 ../tools/compare.py out_s.txt data/expected_s.txt
python3 ../tools/compare.py out_deq.txt data/expected_deq.txt
python3 ../tools/compare.py out_matmul.txt data/expected_matmul_out.txt
python3 ../tools/compare.py out_argmax.txt data/expected_argmax.txt --exact
python3 ../tools/compare.py out_logits.txt data/expected_logits_lastpos.txt
python3 ../tools/compare.py out_ids.txt data/expected_greedy_ids.txt --exact
python3 ../tools/compare.py out_text.txt data/expected_greedy_text.txt --text
```

## 排错路线

1. argmax 全部正确但 logits 有偏差：量化有损，这是正常现象；改为检查贪心文本。
2. 从第一个位置开始 argmax 就错误：quantize 的 scale/round 或 int8 matmul 的分组索引有误；回到 11.2/11.3。
3. 前几层正确、后续逐渐分歧：检查每次 matmul 是否都重新量化了**当前**激活，不要混用 xq/hq 缓冲区。
4. 与 FP32 版本输出不同不一定是错误：量化会改变数值，并可能改变贪心路径。本模型的两个版本会生成相同的 64 步故事，可直接比较验证。

## 完成之后

恭喜，你已经完整重建了 `runq.cpp`。接下来可以：

- 测量 FP32 与 int8 版本的 tok/s，检验“带宽 ÷ 字节数”的估算。
- 阅读 `../tools/dump_fp32.cpp` / `../tools/dump_int8.cpp`，检查自己是否理解每个中间张量。
- 在 `stories42M.bin` / `stories42M-q32.bin` 上重新运行全部内容（使用工具重新生成数据）。
- 最后阅读 `../../runq.cpp` 本身：逻辑相同，但以工程级现代 C++ 方式组织。
