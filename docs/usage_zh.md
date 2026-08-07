# 用法 <span style="float: right;"><a href="usage.md">English</a></span>

```bash
./cpu/runcpp <checkpoint> [options]
```

示例：

```bash
# 贪心解码（确定性输出，temperature=0）
./cpu/runcpp models/stories15M.bin -t 0.0 -n 256 -s 42 -i "Once upon a time"

# 采样解码
./cpu/runcpp models/stories42M.bin -t 0.8 -p 0.9 -n 256 -s 42 -i "One day, a little girl named Lily"

# 对话模式
./cpu/runcpp models/stories15M.bin -m chat -n 256

# 量化模型（用 runqcpp）
./cpu/runqcpp models/stories15M-q32.bin -t 0.0 -n 256 -s 42 -i "Once upon a time"
```

参数说明：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `-t <float>` | temperature，0 = 贪心确定性输出 | 1.0 |
| `-p <float>` | top-p（nucleus sampling），1.0 = 关闭 | 0.9 |
| `-s <int>` | 随机种子 | 当前时间 |
| `-n <int>` | 生成步数，0 = 最大序列长度 | 256 |
| `-i <string>` | 输入 prompt | 空 |
| `-z <string>` | 自定义 tokenizer 路径 | models/tokenizer.bin |
| `-m <string>` | 模式：generate 或 chat | generate |
| `-y <string>` | chat 模式的 system prompt | 无 |

生成结束后 stderr 会打印 `achieved tok/s`，可用于对比 FP32 与 int8 的速度差异。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `cpu/run.cpp` | FP32 推理，主角。前向传播、KV cache、BPE tokenizer、采样全部在这一个文件里 |
| `cpu/runq.cpp` | int8 量化推理。与 `run.cpp` 的差异就是量化：`QuantizedTensor`（int8 + 组缩放因子）、int8 matmul、每次 matmul 前量化激活 |
| `cpu/quantize.cpp` | checkpoint 转换工具：把 FP32 `.bin` 转成 runq 能读的 int8 格式（只需 C++17） |
| `models/tokenizer.bin` | BPE 分词器数据（Llama 2 32K 词表） |
| `models/stories15M.bin` / `models/stories42M.bin` | FP32 模型权重（TinyStories 小模型，来自 [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas)） |
| `models/stories*-q32.bin` | int8 量化权重（由 `quantize` 生成，GS=32） |
