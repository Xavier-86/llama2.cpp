# 推理教程：逐模块构建 llama2 推理 <span style="float: right;"><a href="README.md">English</a></span>

[← 项目首页](../README_zh.md)

> 本教程是 `run.cpp` / `runq.cpp` 的动手实践指南：借助每一步的标准输入输出数据，逐个模块亲手重新实现 int8 量化的 Llama-2 推理。

## 使用方式

每个模块文件夹包含：

- `README.md` / `README_zh.md`：背景、数学原理、输入数据的结构与来源、任务分解与验证方法——只看 README 即可开工
- `main.cpp`：模板，需要完成的任务以 `TODO` 标注
- `solution.cpp`：参考答案，可编译并通过全部对比
- `data.h`（部分模块）：输入测试向量的 const 数组，由 `../tools/embed_data.py` 从 `data/input_*.txt` 生成，不要手改
- `data/`：`input_*.txt`（data.h 的数据源头）和 `expected_*.txt`（标准答案），每行一个数，以 3 位小数输出（`setprecision(3)`）

跨模块共享的样板集中在 `common/`（模块代码只聚焦算法本身）：

- `common/io.h`：标准数据格式的读写（`tut::write_floats` / `write_ints` / `write_text` / `read_floats`）
- `common/checkpoint.h`：FP32 checkpoint 加载器（`tut::load_checkpoint`）——模块 01 答案的复用封装，07/09/11 直接使用
- `common/tokenizer.h`：tokenizer.bin 词表加载（`tut::load_vocab`），02/11/12 使用；BPE 算法本身仍是学习任务

输入数据一律写成 const 变量（main 之外）：小数组直接内联在 `main.cpp`，大数组在各模块的 `data.h`。二进制解析只在模块 01 中专门学习。唯一例外是 10_sampler 的 32000 维真实 logits，太大不内嵌，用 `tut::read_floats` 一行加载。

每个模块的工作流程：

1. 阅读该模块的 `README_zh.md`
2. 补全 `main.cpp`（或自行编写文件），然后构建并运行
3. 将输出与标准数据对比：

```bash
# 浮点数（默认 atol=1e-3、rtol=1e-3，与 3 位小数数据一致）
python3 ../tools/compare.py out.txt data/expected_xxx.txt

# 整数：token id、int8 值（精确比较）
python3 ../tools/compare.py out.txt data/expected_xxx.txt --exact

# 文本：解码结果、生成的故事（逐字节比较）
python3 ../tools/compare.py out.txt data/expected_xxx.txt --text
```

构建命令：`c++ -O2 -std=c++20 -o main main.cpp`

在自己的版本通过测试前，尽量不要阅读 `solution.cpp`（或参考实现 `../cpu/run.cpp` / `../cpu/runq.cpp`）。卡住时可以查看，找到思路后再关掉。

## 模块路线图

| # | 模块 | 实现内容 | 依赖 | 时间 |
| --- | --- | --- | --- | --- |
| 00 | [00_setup](00_setup/README_zh.md) | 构建参考实现，建立基线 | — | 0.5h |
| 01 | [01_checkpoint](01_checkpoint/README_zh.md) | 解析 checkpoint 头部并映射权重（二进制格式的专门小节） | — | 1-2h |
| 02 | [02_tokenizer](02_tokenizer/README_zh.md) | BPE 编码 / 解码 | — | 2-3h |
| 03 | [03_rmsnorm_softmax](03_rmsnorm_softmax/README_zh.md) | 两个小型数学内核 | —（数据已内嵌） | 0.5h |
| 04 | [04_matmul](04_matmul/README_zh.md) | FP32 矩阵-向量乘法 | — | 0.5h |
| 05 | [05_rope](05_rope/README_zh.md) | 旋转位置编码 | —（数据已内嵌） | 1h |
| 06 | [06_attention](06_attention/README_zh.md) | 多头因果注意力 + KV cache | 03、05 | 2h |
| 07 | [07_ffn](07_ffn/README_zh.md) | SwiGLU 前馈网络 | 04（权重经 common/checkpoint.h 获取） | 1h |
| 08 | [08_transformer_layer](08_transformer_layer/README_zh.md) | 组装单个 transformer 层 | 03-07 | 1h |
| 09 | [09_forward](09_forward/README_zh.md) | 完整的单步前向传播（FP32） | 08 | 1h |
| 10 | [10_sampler](10_sampler/README_zh.md) | argmax / temperature / top-p 采样 | 03 | 1-2h |
| 11 | [11_generate](11_generate/README_zh.md) | prefill/decode 生成循环 | 09、10、02 | 1h |
| 12 | [12_quantize](12_quantize/README_zh.md) | int8 量化：格式、内核、前向传播 | 模块 11 的全部内容 | 2-3h |

完成模块 11 表示你已经重建了 `run.cpp`（FP32）；模块 12 加入量化，最终得到 `runq.cpp`。

## 测试设置与数据约定

- 模型：`../../models/stories15M.bin`（FP32）和 `../../models/stories15M-q32.bin`（int8，GS=32）；分词器：`../../models/tokenizer.bin`
- 模型配置：dim=288、hidden_dim=768、n_layers=6、n_heads=6、n_kv_heads=6、vocab_size=32000、seq_len=256（head_size=48、kv_dim=288）
- 参考提示词：`"Once upon a time"` -> token id `[1, 9038, 2501, 263, 931]`，共 P=5 个位置
- 保存各位置数值的文件按**位置优先**拼接：先放 pos 0 的 `dim` 个值，再放 pos 1，以此类推
- 矩阵按**行优先**存储
- 浮点值以 `%.3e` 输出；比较时使用 3 位小数精度

## 数值容差说明

- 小型内核（03/04/05/06/07）：输出应在约 1e-3 内一致（数据本身保留 3 位小数），更大的差异通常意味着存在错误。
- 完整前向传播（09/12）：求和顺序差异（编译器向量化、循环顺序）会造成约 1e-3 的正常波动。先比较 argmax（整数，更稳定），再比较 logits。
- 采样与生成（10/11）是**离散的**：结果要么完全正确，要么错误。若结果错误，从模块 09 的 logits 向前排查。

## 重新生成标准数据

数据由 `tools/dump_fp32.cpp` / `tools/dump_int8.cpp` 生成。这些工具通过 `#include` 引用参考实现并导出中间值，因此标准数据与 `run.cpp` / `runq.cpp` 同源。`data/input_*.txt` 生成后，再运行 `tools/embed_data.py` 把输入向量嵌入为各模块 `data.h` 中的 const 数组。若要针对其他模型或提示词重新生成，请在仓库根目录运行：

```bash
c++ -O2 -std=c++20 -o inference_tutorial/tools/dump_fp32 inference_tutorial/tools/dump_fp32.cpp
c++ -O2 -std=c++20 -o inference_tutorial/tools/dump_int8 inference_tutorial/tools/dump_int8.cpp
./inference_tutorial/tools/dump_fp32 inference_tutorial models/stories15M.bin models/tokenizer.bin
./inference_tutorial/tools/dump_int8 inference_tutorial models/stories15M-q32.bin models/tokenizer.bin
python3 inference_tutorial/tools/embed_data.py inference_tutorial
```

提示词硬编码在 dump 工具中；修改后需重新构建。
