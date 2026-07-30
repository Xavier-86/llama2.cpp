# 11 generate：生成循环（prefill / decode） <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

## 总任务

补全 `main.cpp` 中的 `TODO(task 6)`——`generate()`，把 forward（模块 09）、sampler（模块 10）和 tokenizer（模块 02）串成完整的文本生成流程。这是本模块唯一的新工作；通过它意味着你已重建 FP32 `../../run.cpp`。

前面每个模块都是"一次调用"：一次前向、一次采样。真正的文本生成是一个**循环**——模型每步只预测下一个 token，把这个 token 喂回去，再预测下一个：

```
"Once upon a time"  --encode-->  [1, 9038, 2501, 263, 931]  --prefill / decode 循环-->  ids + text
```

循环有两个阶段，对应服务系统里的两个核心指标（TTFT ≈ prefill 时间，吞吐量 ≈ decode 速度）：**prefill** 逐个把提示词 token 强制跑过模型、填满 KV cache；**decode** 随后每步采样一个 token，串行进行。

代码骨架已经搭好：checkpoint 解析（模块 01 的产物）由 `../common/checkpoint.h` 的 `tut::load_checkpoint` 提供，词表加载由 `../common/tokenizer.h` 的 `tut::load_vocab` 提供，输出写文件由 `../common/io.h` 提供。你在模块 02–10 中实现的组件在 `main.cpp` 中均为**给定代码**，已按参考实现填好：

- `rmsnorm` / `softmax` / `matmul`——模块 03/04：rmsnorm 累加用 float、softmax 先减最大值、matmul 用 float 累加器。
- `Transformer::forward(token, pos)`——模块 09：单 token 单位置的完整前向；K/V 写入本层本位置的缓存行后再做 RoPE；残差是 `+=`。
- `Tokenizer::decode` / `Tokenizer::encode`——模块 02：在构造函数就绪的词表（`vocab_` / `vocab_scores_`）之上的 BPE 算法。
- `Sampler`——模块 10：`temperature == 0` 走 argmax；否则温度缩放 + softmax + 一次掷币，再按 `topp` 走全分布或 top-p。RNG 与参考逐位一致。

**输入**：不需要自己解析任何文件；输入以下来源都已就绪：

| 变量 | 位置 | 形状 / 布局 | 含义 | 来自模型哪里 |
| --- | --- | --- | --- | --- |
| `prompt` | main() 内联 const 字符串 | `"Once upon a time"` | 提示词，encode 后为 `[1, 9038, 2501, 263, 931]`（P=5，含 BOS） | 用户输入 |
| `transformer.checkpoint.config` | `tut::load_checkpoint("../../stories15M.bin")` | 7 个 int32 | 模型超参数 | checkpoint 头部 |
| `transformer.checkpoint.weights` | 同上 | 11 个 `std::span<const float>` | 全部权重张量（见下） | checkpoint 权重区 |
| `tokenizer`（构造参数） | `tut::load_vocab("../../tokenizer.bin", vocab_size)` | `pieces`(32000,) / `scores`(32000,) | token id → 文本片段 / BPE 合并分数 | tokenizer.bin |
| `steps` | main() 内联 | 64 | 生成步数上限 | — |
| temperature / topp / seed | main() 内联 | 见下方运行表 | 两组采样参数 | — |

模型常量（stories15M）：`dim = 288`，`hidden_dim = 768`，`n_layers = 6`，`n_heads = 6`，`n_kv_heads = 6`，`vocab_size = 32000`，`seq_len = 256`，`head_size = dim / n_heads = 48`，`kv_dim = n_kv_heads * head_size = 288`。

`weights` 的 11 个张量（shape 用上述常量表示；checkpoint 中按此顺序紧密排列，行主序）：

| 张量 | 形状 | 用途 |
| --- | --- | --- |
| `token_embedding_table` | (vocab_size, dim) | embedding 查表；本模型 wcls 与之共享 |
| `rms_att_weight` | (n_layers, dim) | 注意力前 RMSNorm 权重 |
| `wq` / `wk` / `wv` | (n_layers, dim, dim) / (n_layers, dim, kv_dim) ×2 | Q/K/V 投影 |
| `wo` | (n_layers, dim, dim) | 注意力输出投影 |
| `rms_ffn_weight` | (n_layers, dim) | FFN 前 RMSNorm 权重 |
| `w1` / `w3` | (n_layers, hidden_dim, dim) ×2 | FFN 上行投影（gate / linear） |
| `w2` | (n_layers, dim, hidden_dim) | FFN 下行投影 |
| `rms_final_weight` | (dim,) | 最终 RMSNorm 权重 |
| `wcls` | (vocab_size, dim) | 分类头；共享时别名 embedding 表 |

另外 `RunState` 在构造时按 config 分配好激活缓冲与 KV cache（`key_cache` / `value_cache` 各为 `(n_layers, seq_len, kv_dim)`），这也是输入的一部分——每步 forward 在其中读写。

**输出**：`main()` 已写好——加载 checkpoint 与词表，跑贪心与采样两组，写出 4 个文件。文本文件写出时追加一个尾部换行（`g.text + '\n'`），与 golden 数据的约定一致。`data/expected_*` 是黄金数据，不要修改。

`main()` 固定跑两组配置（提示词均为 `"Once upon a time"`，`steps = 64`）：

| 运行 | temperature | top-p | seed | 输出文件 |
| --- | --- | --- | --- | --- |
| 贪心（greedy） | 0.0（argmax 路径，不消耗随机数） | 0.9（未使用） | 42 | `out_ids.txt` / `out_text.txt` |
| 采样（sampled） | 0.8 | 0.9 | 42 | `out_sids.txt` / `out_stext.txt` |

## 子任务一：生成循环 `generate()`（`TODO(task 6)`）

实现 prefill/decode 循环，返回 `Generation{ids, text}`：`ids` 是循环产生的每个 `next`——含 prefill 阶段强制输入的 4 个，不含 BOS 本身；`text` 是 `decode(token, next)` 片段的拼接。按顺序：

1. `prompt_tokens = encode(prompt, bos=true, eos=false)`；初始 `token = prompt_tokens[0]`（BOS），`pos = 0`。
2. 当 `pos < steps`：执行 `logits = forward(token, pos)`，然后决定下一步输入 `next`——prefill 或 decode，见下。
3. `pos += 1`；若 `next == 1` 则停止；否则把 `next` 加入 `ids`，把 `decode(token, next)` 拼到 `text`，再令 `token = next`。

需要的知识——prefill 与 decode 两阶段。设提示词编码后得到 `prompt_tokens`（长度 P=5：`[1, 9038, 2501, 263, 931]`，其中 1 是 BOS `<s>`）。循环每一步做一次 `forward(token, pos)`，然后决定下一步的输入 `next`：

- **prefill（预填充）**：`pos < P - 1` 时，`next` 不来自模型，而是**强制取自提示词**：`next = prompt_tokens[pos + 1]`。这一阶段逐个把提示词 token 跑过模型，目的是把它们的 K/V 填进 KV cache，同时得到"提示词最后一位置"的 logits。不采样、不消耗随机数。
- **decode（解码）**：提示词消耗完后（`pos >= P - 1`），`next = sample(logits)`——由模型（经采样器）决定。之后每步都是：采样一个 token → 喂回模型 → 再采样。串行，无法并行，每生成一个词都要完整 forward 一次。

整个循环的伪代码：

```
prompt_tokens = encode(prompt, bos=true)      # [1, 9038, 2501, 263, 931]
token = prompt_tokens[0];  pos = 0
while pos < steps:                            # steps = 64
    logits = forward(token, pos)
    if pos < len(prompt_tokens) - 1:
        next = prompt_tokens[pos + 1]         # prefill：强制输入下一个提示词 token
    else:
        next = sample(logits)                 # decode：由模型决定
    pos += 1
    if next == 1: break                       # 再次出现 BOS(id=1) = 停止
    ids.push_back(next)
    text += decode(token, next)               # 先 decode(token, next)，再 token = next
    token = next
```

注意三个细节：

1. `pos` 每步递增，既是位置编码（RoPE）的输入，也是 KV cache 的写入行号。
2. 停止条件是 `next == 1`（模型再次生成 BOS）；`steps` 只是上限。
3. 先 `decode(token, next)`（需要"前一个 token"来判断是否剥离前导空格），再 `token = next`。

需要的知识——KV cache 在循环中的作用。`forward(token, pos)` 每步只为**当前这一个 token** 计算 K/V，写入 `key_cache[layer][pos]` / `value_cache[layer][pos]`；注意力读取的是第 `0..pos` 行——历史位置的 K/V 全部来自缓存，不重算。prefill 阶段就是在为 decode 阶段"预热"这个缓存：没有 prefill 填进去的前 P 行，decode 第一步的注意力就没有上下文可看。这也是为什么 decode 每步的计算量与序列位置无关（一个 token 的前向），只有注意力的历史长度随 `pos` 线性增长。

需要的知识——RNG 的可复现性。采样运行的可复现性完全依赖 RNG：每步恰好消耗一个随机数（xorshift64，见模块 10），顺序必须与参考实现一致——多调用一次，从第一步起全部分歧。所以每个 decode 步必须恰好调用一次 `sample(logits)`，prefill 阶段则一次都不调。

## 构建 / 运行 / 验证

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out_ids.txt   data/expected_greedy_ids.txt --exact
python3 ../tools/compare.py out_text.txt  data/expected_greedy_text.txt --text
python3 ../tools/compare.py out_sids.txt  data/expected_sampled_ids.txt --exact
python3 ../tools/compare.py out_stext.txt data/expected_sampled_text.txt --text
```

## 常见错误

- 贪心 PASS 但采样 FAIL → 检查模块 10 的 sampler（温度缩放、top-p 截断、RNG）。
- 两组都错但前几个 id 正确 → 检查 prefill/decode 的切换条件（`pos < num_prompt_tokens - 1`）。
- 采样从第一个 decode token 就分歧 → 每步多消耗了随机数：全循环每步恰好一次掷币。
- id 全对但文本 FAIL → `decode` 的参数顺序应为 `(token, next)`（当前 token 在前）；或漏了 BOS 后的前导空格剥离；或写文件时漏了尾部换行。
- id 序列前 4 个不是 `[9038, 2501, 263, 931]` → prefill 阶段没有强制取 `prompt_tokens[pos + 1]`。

## 完成标准

四项比较全部 PASS（两个 `--exact`，两个 `--text`）。可选：按 task 6 的提示计时，tok/s 与参考实现的 `achieved tok/s` 应在同一量级。`solution.cpp` 是参考答案——卡住再看，看完关掉自己写。至此 FP32 全部完成：运行 `./main`，看自己的实现讲出与原版相同的故事。
