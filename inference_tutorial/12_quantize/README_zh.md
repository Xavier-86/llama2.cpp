# 12 quantize：int8 量化推理 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

## 总任务

补全 `main.cpp` 中四组量化 TODO（task 1a/1b、2a/2b、3、4——编号与代码一一对应）：将完成的 FP32 实现升级为 int8 量化并重建 `runq.cpp`。改动集中在三个部分：权重存储格式、矩阵乘法，以及每次矩阵乘法前对激活进行量化。其余部分（tokenizer、sampler 和生成循环）均原样复用——BPE tokenizer（模块 02）、sampler（模块 10）、generate/chat 循环（模块 11）、rmsnorm/softmax（模块 03），以及 mmap 样板（`MappedFile`）在 `main.cpp` 中均为**给定**；词表解析走 `../common/tokenizer.h` 的 `tut::load_vocab`。

量化为什么更快：每个 decode 步骤都会读取一次全部权重；速度 ≈ 带宽 ÷ 字节数。int8 将权重读取量降至 1/4（每 32 个值额外带一个 float scale），因此吞吐量可提升近 4 倍——这就是本模块的全部动机。

**输入**：两个真实文件（mmap 的量化 checkpoint、词表）加一批内嵌 const 数组：

| 变量 | 位置 | 形状 / 布局 | 含义 | 来自模型哪里 |
| --- | --- | --- | --- | --- |
| `checkpoint_path` | main.cpp harness（`../../models/stories15M-q32.bin`） | 256 字节头 + FP32/int8 混合权重区（详见子任务一） | int8 量化 checkpoint，`MappedFile` 只读 mmap | 由仓库根目录的 `quantize` 工具从 `stories15M.bin` 导出（GS=32） |
| `tokenizer_path` | main.cpp harness（`../../models/tokenizer.bin`） | [i32 max_token_length] + 32000 条 [f32 score][i32 len][len 字节 piece] | BPE 词表，由 `tut::load_vocab` 解析为 `tut::Vocab` | 与所有 FP32 模块相同 |
| `kX` | data.h | (64,)，两组各 32 个 float | 子任务二 quantize/dequantize 往返的测试向量 | 教学合成数据（镜像 data/input_x.txt） |
| `kMatmulWQ` | data.h | (3, 8) 行优先，24 个 int8 | 子任务三独立 int8 matmul 的量化权重 | 教学合成数据（此夹具故意用 GS=4） |
| `kMatmulWS` | data.h | (3, 2) 行优先，6 个 float | w 的逐组 scale | 同上 |
| `kMatmulXQ` | data.h | (8,) int8 | 量化输入向量 x | 同上 |
| `kMatmulXS` | data.h | (2,) float | x 的逐组 scale | 同上 |
| `kTokens` | main.cpp harness | (5,) int = {1, 9038, 2501, 263, 931} | 子任务四五位置前向检查的提示词 token id | "Once upon a time" 经 BPE（含 bos）编码 |

模型常量（stories15M）：dim=288，hidden_dim=768，n_layers=6，n_heads=6，n_kv_heads=6（kv_dim=288），vocab_size=32000，seq_len=256，head_size=48；本量化 checkpoint 的 GS=32、shared_classifier=1。

`data.h` 由 `../tools/embed_data.py` 从 `data/*.txt` 生成，勿手改——值与 txt 完全一致，重新生成数据后跑一遍 `python3 ../tools/embed_data.py ..` 即可。

**输出**：`main()` 已经写好——输出分阶段排错所需的全部中间结果：

| 输出 | 标准数据 |
| --- | --- |
| `out_config.txt` | `expected_config.txt` |
| `out_summary.txt` | `expected_weight_summary.txt` |
| `out_q.txt`、`out_s.txt`、`out_deq.txt` | quantize/dequantize 输出 |
| `out_matmul.txt` | 独立 int8 matmul 输出 |
| `out_argmax.txt`、`out_logits.txt` | 5 个位置的量化前向传播 |
| `out_ids.txt`、`out_text.txt` | 64 步贪心生成 |

`data/expected_*` 是黄金数据，不要修改。

## 子任务一：加载并映射量化权重——`init_quantized_tensors()` / `Transformer::map_weights()`（task 1a/1b）

- **task 1a — `init_quantized_tensors`**：从字节指针切出 n 个量化张量，每个为 `[size_each 个 int8][size_each/GS 个 float scale]`，每切完一个 q 区和一个 s 区就前移指针。
- **task 1b — `Transformer::map_weights`**：按下面 checkpoint 的顺序映射 FP32 区（rms_att / rms_ffn / rms_final）和 int8 区（q_tokens、wq、wk、wv、wo、w1、w2、w3，可选 wcls），并把 q_tokens 反量化出 FP32 副本；shared 时让 wcls 直接别名 q_tokens[0]。

需要的知识——量化 checkpoint 格式（`stories15M-q32.bin`）：

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

与 FP32 checkpoint（`stories15M.bin`，见模块 01/11）的差异：

| 部位 | FP32 版 | 量化版 |
| --- | --- | --- |
| 头部 | 7 个 i32 的 Config（负 vocab_size 表示 wcls 不共享） | 固定 256 字节：magic / version / Config / shared 标志 / GS |
| rms_att / rms_ffn / rms_final 权重 | float | **仍是 float**——一维小向量，量化收益小、精度损失大 |
| token_embedding_table | float | int8 + scales（加载时反量化出 float 副本供查表） |
| wq / wk / wv / wo / w1 / w2 / w3 | float | int8 + scales |
| wcls | float，可共享 embedding | int8，可共享 embedding（本文件 shared=1） |
| 遗留的 freq_cis_real / freq_cis_imag 表 | 存在（加载时跳过） | 不存在 |

mmap 区域怎么切（本文件总大小 17,101,696 字节，可对照验证）：`MappedFile::weights_data()` 返回偏移 256；`map_weights` 按上表顺序依次切 span——先切 3 个 FP32 张量，再对每个量化张量调用 `init_quantized_tensors`（先切 int8 值区，再切 float scales 区）：

```
偏移（字节）  内容                                            大小
0            u32 magic = 0x616b3432 ("ak42")                 4
4            i32 version = 2                                 4
8            Config（7 × i32）                               28
36           u8 shared_classifier = 1                        1
37           i32 group_size (GS) = 32                        4
41           （未使用，头部固定 256 字节）
256          rms_att_weight   6×288 float                    6,912
7,168        rms_ffn_weight   6×288 float                    6,912
14,080       rms_final_weight 288 float                      1,152
15,232       q_tokens.q       9,216,000 int8                 9,216,000
9,231,232    q_tokens.s       288,000 float                  1,152,000
10,383,232   wq[0].q/s … wq[5].q/s（每层 82,944 int8 + 2,592 float）   559,872
10,943,104   wk（布局同 wq）                                  559,872
11,502,976   wv（布局同 wq）                                  559,872
12,062,848   wo（布局同 wq）                                  559,872
12,622,720   w1[0..5]（每层 221,184 int8 + 6,912 float）      1,492,992
14,115,712   w2（布局同 w1）                                  1,492,992
15,608,704   w3（布局同 w1）                                  1,492,992
17,101,696   文件结束（shared=1，没有 wcls 区）
```

对照 FP32 版的 60,816,028 字节，文件缩小约 3.6 倍；每个 decode 步的权重读取量同比例下降。

本子任务的验证方法：输出 config + GS（8 个整数），用 `--exact` 与 `expected_config.txt` 比较；再输出类似模块 01 的权重摘要。`expected_weight_summary.txt` 先包含 3 个 FP32 张量各 3 行（size/first/sum），再包含 8 个量化张量各 5 行（q_total_size/q_first/q_sum/s_total_size/s_first）。命令见下文"构建 / 运行 / 验证"。

## 子任务二：`quantize()` / `dequantize()`（task 2a/2b）

- **task 2a — `dequantize`**：`x[i] = q[i] * s[i/GS]`。
- **task 2b — `quantize`**：每组 GS 个值，`scale = max|group| / 127`，`q[i] = std::round(x[i] / scale)`。

两者都从 span 长度之比推出 GS，不要写死 32。

需要的知识——对每组 GS 个值进行对称量化：

```
scale = max(|group values|) / 127
q[i]  = round(x[i] / scale)     # int8，范围 [-127, 127]
s[g]  = scale
dequantize: x[i] = q[i] * s[i / GS]
```

数据：`kX`（data.h，64 个值 = 两组 32，GS=32）-> `expected_q.txt`（64 个 int8，使用 `--exact`）、`expected_s.txt`（2 个 scale）、`expected_deq.txt`（往返结果）。内核不写死 GS，而是从 `QuantizedTensor` 两个 span 的长度之比推出来（子任务三的夹具用的是 GS=4）。

注意：这里的 `round` 是中点值**远离零**舍入（`std::round`），不是银行家舍入。尤其是用 Python 交叉验证时要注意，Python 的 `round` 使用银行家舍入。

## 子任务三：int8 `matmul()`（task 3）

**task 3 — `matmul`**：W(d,n) @ x(n)，每组累加进 int32，组边界处加 `group_sum * w.s * x.s` 进 float 行结果，并在组边界清零整数累加器。

需要的知识——分组算法：

```
对每一行：
    按 GS 大小分组：int8 x int8 -> int32 整数乘加
    （不会溢出：127*127*32 ~ 5*10^5）
    每组结束时：val += (float)group_int_sum * w.s[group] * x.s[group]
```

整数部分精确计算，只有组边界处出现浮点运算，因此量化版本既快又稳定。

数据（n=8、d=3、GS=4，全部在 data.h）：`kMatmulWQ`（(3,8) 行优先 24 个 int8）、`kMatmulWS`（(3,2) 行优先 6 个 scale）、`kMatmulXQ`（8 个 int8）、`kMatmulXS`（2 个 scale）-> `expected_matmul_out.txt`（3 个值）。

## 子任务四：量化版 `Transformer::forward()`（task 4）

**task 4 — `Transformer::forward`**：移植模块 09 的前向传播，在每次 matmul 前量化**当前**激活（插入点为下面五处）；RoPE、attention、残差、RMSNorm、SwiGLU 保持 FP32。

需要的知识——量化插在哪里。基于模块 09 的 forward，在**每次 matmul 之前**将激活量化为 int8，再通过子任务三的 int8 权重执行乘法。RMSNorm、RoPE、attention、SwiGLU、softmax 均继续使用 FP32，不做修改：

```
rmsnorm xb      -> quantize xq -> Wq / Wk / Wv
attention 输出 xb -> quantize xq -> Wo
rmsnorm xb      -> quantize xq -> W1 / W3
SwiGLU hb       -> quantize hq -> W2
final rmsnorm x -> quantize xq -> Wcls
```

数据：`kTokens`（与此前模块相同的 5 个 token）-> `expected_argmax.txt`（5 个整数，使用 `--exact`）+ `expected_logits_lastpos.txt`（最后位置的完整 32000 维 logits）。最后运行 64 步贪心生成，与 `expected_greedy_ids.txt` / `expected_greedy_text.txt` 比较。

## 构建 / 运行 / 验证

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

## 常见错误

1. argmax 全部正确但 logits 有偏差：量化有损，这是正常现象；改为检查贪心文本。
2. 从第一个位置开始 argmax 就错误：quantize 的 scale/round 或 int8 matmul 的分组索引有误；回到子任务二、三。
3. 前几层正确、后续逐渐分歧：检查每次 matmul 是否都重新量化了**当前**激活，不要混用 xq/hq 缓冲区。
4. 与 FP32 版本输出不同不一定是错误：量化会改变数值，并可能改变贪心路径。本模型的两个版本会生成相同的 64 步故事，可直接比较验证。
5. 权重摘要从某一项开始全部错位：`init_quantized_tensors` 的指针前移漏了 q 区或 s 区——一个张量错，后面全错。

## 完成标准

上面 10 条 compare.py 全部 PASS。`solution.cpp` 是参考答案——卡住再看，看完关掉自己写。

## 完成之后

恭喜，你已经完整重建了 `runq.cpp`。接下来可以：

- 测量 FP32 与 int8 版本的 tok/s，检验“带宽 ÷ 字节数”的估算。
- 阅读 `../tools/dump_fp32.cpp` / `../tools/dump_int8.cpp`，检查自己是否理解每个中间张量。
- 在 `stories42M.bin` / `stories42M-q32.bin` 上重新运行全部内容（使用工具重新生成数据）。
- 最后阅读 `../../cpu/runq.cpp` 本身：逻辑相同，但以工程级现代 C++ 方式组织。
