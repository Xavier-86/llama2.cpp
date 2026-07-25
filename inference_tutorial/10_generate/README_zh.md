# 10 generate：生成循环（prefill / decode） <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md)

> 目标：将 forward、sampler 和 tokenizer 串成完整的文本生成流程。通过本模块意味着你已重建 FP32 `run.cpp`。

## 循环结构

```
prompt_tokens = encode(prompt, bos=true)
token = prompt_tokens[0];  pos = 0
while pos < steps:
    logits = forward(token, pos)
    if pos < len(prompt_tokens) - 1:
        next = prompt_tokens[pos + 1]   # prefill：强制输入下一个提示词 token
    else:
        next = sample(logits)           # decode：由模型决定
    pos += 1
    if next == 1: break                 # 再次出现 BOS = 停止
    print decode(token, next)           # 注意传入（当前 token, 下一个 token）
    token = next
```

- **prefill**：逐 token 运行提示词，填充 KV cache，不进行采样。
- **decode**：每步采样一个 token 并反馈给模型；过程串行，每生成一个词都需完整 forward 一次。
- 服务系统中的 TTFT ≈ prefill 时间，吞吐量 ≈ decode 速度：分别对应循环的两个阶段。

## 数据文件（提示词始终为 `"Once upon a time"`，64 步）

| 文件 | 内容 |
| --- | --- |
| `expected_greedy_ids.txt` | 贪心解码（t=0）的 token id |
| `expected_greedy_text.txt` | 贪心生成文本 |
| `expected_sampled_ids.txt` | 采样（t=0.8、p=0.9、seed=42）的 id |
| `expected_sampled_text.txt` | 采样生成文本 |

## 任务与验证

```bash
python3 ../tools/compare.py out_ids.txt data/expected_greedy_ids.txt --exact
python3 ../tools/compare.py out_text.txt data/expected_greedy_text.txt --text
python3 ../tools/compare.py out_sids.txt data/expected_sampled_ids.txt --exact
python3 ../tools/compare.py out_stext.txt data/expected_sampled_text.txt --text
```

贪心正确但采样错误：检查模块 09 的 sampler。两者都错但前几个 id 正确：检查 prefill/decode 的切换条件。

## 提示

- id 序列记录循环产生的每个 `next` token，因此前四个 id 是强制输入的提示词后续部分 `[9038, 2501, 263, 931]`；不包含 BOS（`1`）本身。
- 采样运行每步恰好消耗一个随机数，顺序必须与参考实现完全一致；多调用一次 RNG 就会从第一步开始分歧。
- 加入计时：从第二个 token 开始测量 tok/s，并与参考实现的 `achieved tok/s` 比较，预期处于同一量级。
- 至此 FP32 已完成。运行 `./main`，观察自己的实现讲出与原版相同的故事。
