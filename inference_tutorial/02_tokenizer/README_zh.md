# 02 tokenizer：BPE 编码与解码 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md)

> 目标：实现 `encode`（文本 -> token id）和 `decode`（id -> 文本片段）。这是独立模块，不需要模型权重，只需 `../../tokenizer.bin`。

## tokenizer.bin 格式

```
[max_token_length: int32]
随后是 vocab_size（=32000）个条目，每个条目：
  [score: float32] [len: int32] [长度为 len 字节的字符串]
```

词表约定：

- id 0 = `<unk>`，id 1 = `<s>`（BOS），id 2 = `</s>`（EOS）
- id 3..258 是 256 个单字节 token（`<0x00>`..`<0xFF>`）；字节 b 对应的 token id 为 **b + 3**
- 其余均为 BPE 合并后的子词；`score` 越高，合并优先级越高

## encode 算法（与参考实现中的 `Tokenizer::encode` 一致）

输入为 `text` 以及 `bos` / `eos` 开关（本项目中 eos 始终为 false）：

1. 若 `bos=true`，先加入 id 1。
2. 若文本非空，加入 `" "`（空格）的 id。Llama 约定按文本前带空格的方式分词。
3. **逐字符查找**：按 UTF-8 边界切分（满足 `(c & 0xC0) != 0x80` 的字节是新字符的起点）；在词表中查找每个字符；若不存在，则退化为逐字节 token（`byte + 3`）。
4. **贪心合并**：反复扫描相邻 token 对；若其拼接字符串存在于词表，则合并分数最高的一对；没有可合并项时停止。

## decode 算法

`decode(prev_token, token)` 返回该 token 的字符串，并应用两条规则：

- 若 `prev_token == 1`（前一个是 BOS）且片段以空格开头，去掉该空格。
- 形如 `<0xXX>`（6 个字符）的片段解码为对应字节。

## 数据文件

| 文件 | 内容 |
| --- | --- |
| `input_prompts.txt` | 4 行，每行一个提示词 |
| `expected_encode_0..3.txt` | token id 序列（包含 BOS） |
| `input_decode_ids.txt` | 提示词 0 的 id 序列 |
| `expected_decode.txt` | 依次拼接 `decode(id[i], id[i+1])` 得到的文本 |

## 任务与验证

```bash
# encode：比较每个提示词的 id
python3 ../tools/compare.py out0.txt data/expected_encode_0.txt --exact
# decode：逐字节比较文本
python3 ../tools/compare.py out_decode.txt data/expected_decode.txt --text
```

## 提示

- 使用哈希表或排序后二分查找加速词表查询。线性扫描虽然可用，但会拖慢合并循环。
- 对 5～10 个 token 的提示词，朴素 O(n²) 合并循环足够快。
- 最常见的两个错误：忘记开头的 `" "` token；合并时查找 id 对而不是拼接后的字符串。
