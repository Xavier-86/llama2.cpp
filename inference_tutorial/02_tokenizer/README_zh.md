# 02 tokenizer：BPE 编码与解码 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

> 目标：实现 BPE 分词器的 `encode`（文本 -> token id）和 `decode`（id -> 文本片段）。这是独立模块，不需要模型权重，只需 `../../tokenizer.bin`。

## 背景

模型看到的不是文字，而是整数序列。tokenizer 负责两个方向的转换：

```
"Once upon a time"  --encode-->  [1, 9038, 2501, 263, 931]  --forward-->  logits  --sample-->  下一个 id  --decode-->  文字
```

本模块实现的 BPE（byte-pair encoding）算法与 `../../run.cpp` 中的 `Tokenizer` 完全一致：先把文本拆成单字符 token，再按词表分数贪心合并相邻对。词表文件解析已由 `../common/tokenizer.h` 代劳，你只需要写算法。

## tokenizer.bin 格式

```
[max_token_length: int32]
随后是 vocab_size（=32000）个条目，每个条目：
  [score: float32] [len: int32] [长度为 len 字节的字符串]
```

解析由 `tut::load_vocab(path, vocab_size)`（`../common/tokenizer.h`）完成，返回 `tut::Vocab{pieces, scores, max_token_length}`：`pieces[id]` 是 token 的文本片段，`scores[id]` 是 BPE 合并分数。**解析不是本模块的学习任务**，构造 `Tokenizer` 时一行调用即可。

词表约定：

- id 0 = `<unk>`，id 1 = `<s>`（BOS），id 2 = `</s>`（EOS）
- id 3..258 是 256 个单字节 token（`<0x00>`..`<0xFF>`）；字节 b 对应的 token id 为 **b + 3**
- 其余均为 BPE 合并后的子词；`score` 越高，合并优先级越高

## encode 算法

输入为 `text` 以及 `bos` / `eos` 开关（本项目中 eos 始终为 false）：

1. 若 `bos=true`，先加入 BOS id 1。
2. 若文本非空，加入 `" "`（空格）的 id。Llama 约定按"文本前带一个空格"的方式分词。
3. **逐字符查找**：按 UTF-8 边界切分——满足 `(c & 0xC0) != 0x80` 的字节是一个新字符的起点（0x80..0xBF 是续字节）；在词表中查找每个字符；若不存在，则退化为逐字节 token（`byte + 3`，即 byte fallback）。
4. **贪心合并**：反复扫描所有相邻 token 对，把它们的字符串拼接后在词表中查找；在能找到的对中，合并 `score` 最高的一对；没有任何对可合并时停止。
5. 若 `eos=true`，末尾加入 EOS id 2。

## decode 算法

`decode(prev_token, token)` 返回该 token 的字符串，并应用两条规则：

- 若 `prev_token == 1`（前一个是 BOS）且片段以空格开头，去掉该空格（与 encode 步骤 2 互为逆操作）。
- 形如 `<0xXX>`（恰好 6 个字符）的片段展开为对应的那一个字节。

## 输入数据

所有输入都是 main.cpp 里的 const 变量（词表除外），无需读任何数据文件：

| 变量 | 位置 | 形状 | 含义 |
| --- | --- | --- | --- |
| `kPrompts` | main.cpp | (4,) 字符串数组 | 4 条待编码的提示词，见下表 |
| `kDecodeIds` | main.cpp | (5,) int 数组 | 待解码的 id 序列，即提示词 0 的 token（含 BOS） |
| `vocab_.pieces` / `vocab_.scores` | `tut::load_vocab` | (32000,) | 词表：token id -> 文本片段 / BPE 合并分数，来自 `../../tokenizer.bin` |

4 条提示词及其预期 token id（含 BOS，即 `data/expected_encode_*.txt` 的内容）：

| # | 提示词 | 预期 token id |
| --- | --- | --- |
| 0 | `Once upon a time` | `[1, 9038, 2501, 263, 931]` |
| 1 | `One day, a little girl named Lily` | `[1, 3118, 2462, 29892, 263, 2217, 7826, 4257, 365, 2354]` |
| 2 | `Hello, world!` | `[1, 15043, 29892, 3186, 29991]` |
| 3 | `The capital of France is` | `[1, 450, 7483, 310, 3444, 338]` |

提示词 0 是整个教程的参考提示词。`data/input_prompts.txt` 和 `data/input_decode_ids.txt` 保留作参考，内容与 const 数组一致；`data/expected_*` 是黄金数据，不要修改。

## 任务

补全 `main.cpp` 中 `Tokenizer` 类的四个 TODO：

1. **task 1 — `init_sorted_vocab()` / `str_lookup(str)`**：实现"字符串 -> token id"的查找：建一次排序索引（或哈希表），之后二分查找；找不到返回 -1。线性扫描也能过，但会拖慢合并循环。
2. **task 2 — `encode` 前半**（算法步骤 1-3）：BOS、前导 `" "`、按 UTF-8 边界逐字符查找 + byte fallback。
3. **task 3 — `encode` 后半**（算法步骤 4-5）：贪心合并 + 可选 EOS。
4. **task 4 — `decode(prev_token, token)`**：返回 token 片段，处理 BOS 后去前导空格和 `<0xXX>` 字节展开。

`main()` 已经写好：它编码 4 条提示词写出 `out0.txt .. out3.txt`，再解码 `kDecodeIds` 写出 `out_decode.txt`。

## 构建 / 运行 / 验证

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out0.txt data/expected_encode_0.txt --exact
python3 ../tools/compare.py out1.txt data/expected_encode_1.txt --exact
python3 ../tools/compare.py out2.txt data/expected_encode_2.txt --exact
python3 ../tools/compare.py out3.txt data/expected_encode_3.txt --exact
python3 ../tools/compare.py out_decode.txt data/expected_decode.txt --text
```

## 常见错误

- 输出比预期多出一串单字符 id → 忘记在 BOS 后加 `" "` token，或合并循环完全没生效。
- 合并结果与预期不同 → 在词表里查的是 id 对而不是**拼接后的字符串**；或者没有选 score 最高的对。
- 多字节字符（中文等）被切碎 → UTF-8 边界判断有误：续字节满足 `(c & 0xC0) == 0x80`，不能当作新字符起点。
- `out_decode.txt` 开头多一个空格 → 忘记处理 `prev_token == 1` 时去掉前导空格。

## 完成标准

五项比较全部 PASS。`solution.cpp` 是参考答案——卡住再看，看完关掉自己写。
