# llama2.cpp：纯 C++ 最小 LLM 推理 <span style="float: right;"><a href="README.md">English</a></span>

> [karpathy/llama2.c](https://github.com/karpathy/llama2.c) 的现代 C++（C++20）重写版，用于学习 LLM 推理原理。
> 与原版差异：RAII 内存管理（`std::vector` / mmap 封装类）、`std::span` 视图传参、`std::string` / `std::string_view`、异常处理、iostream、`std::sort`/`std::lower_bound`。数值行为与原版等价，但代码结构不追求逐行对应。

## 目录

- [文件说明](#文件说明)
- [编译](#编译)
- [运行](#运行)
- [量化模型](#量化模型)
- [分步推理教程](#分步推理教程)
- [调试](#调试)

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `run.cpp` | FP32 推理，主角。前向传播、KV cache、BPE tokenizer、采样全部在这一个文件里 |
| `runq.cpp` | int8 量化推理。与 `run.cpp` 的差异就是量化：`QuantizedTensor`（int8 + 组缩放因子）、int8 matmul、每次 matmul 前量化激活 |
| `quantize.cpp` | checkpoint 转换工具：把 FP32 `.bin` 转成 runq 能读的 int8 格式 |
| `tokenizer.bin` | BPE 分词器数据（Llama 2 32K 词表） |
| `stories15M.bin` / `stories42M.bin` | FP32 模型权重（TinyStories 小模型，来自 [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas)） |
| `stories*-q32.bin` | int8 量化权重（由 `quantize` 生成，GS=32） |
| [`inference_tutorial/`](inference_tutorial/README_zh.md) | 将 FP32 与 int8 推理分解为 12 个可独立验证模块的动手教程 |

注意：`stories*.bin` 权重**不在 git 仓库里**（体积太大）。从 [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) 下载 `stories15M.bin` / `stories42M.bin` 放到本目录即可；`tokenizer.bin` 已包含在仓库中。

## 编译

编译器要求：`run.cpp` / `runq.cpp` 用了 C++20 特性（`std::span`、`std::ranges`、`std::from_chars` 等），需要支持 C++20 的编译器：

- **macOS**：系统自带 `/usr/bin/clang++`（Apple Clang 15+，装 Xcode Command Line Tools 即可，无需另装编译器）。本目录在 Apple Clang 17 上开发验证。
- **其他平台**：GCC 11+ 或 Clang 14+。

注意 `run.cpp` 和 `runq.cpp` 还依赖 POSIX 的 `mmap`（`<sys/mman.h>`），Windows 上不能直接用 MSVC 编译。`quantize.cpp` 只需 C++17。

```bash
c++ -O3 -std=c++20 -o runcpp run.cpp        # FP32 推理
c++ -O3 -std=c++20 -o runqcpp runq.cpp      # int8 量化推理
c++ -O3 -std=c++17 -o quantize quantize.cpp # 量化转换工具
```

用 VSCode 时，IntelliSense 与构建任务的标准版本在 `.vscode/c_cpp_properties.json` 和 `.vscode/tasks.json` 里单独配置（当前已设为 `-std=c++20`）；如果改了这里用的标准，两边要一起改，否则编辑器会误报找不到 `std::span` 等错误。

## 运行

```bash
./runcpp <checkpoint> [options]
```

示例：

```bash
# 贪心解码（确定性输出，temperature=0）
./runcpp stories15M.bin -t 0.0 -n 256 -s 42 -i "Once upon a time"

# 采样解码
./runcpp stories42M.bin -t 0.8 -p 0.9 -n 256 -s 42 -i "One day, a little girl named Lily"

# 对话模式
./runcpp stories15M.bin -m chat -n 256

# 量化模型（用 runqcpp）
./runqcpp stories15M-q32.bin -t 0.0 -n 256 -s 42 -i "Once upon a time"
```

参数说明：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `-t <float>` | temperature，0 = 贪心确定性输出 | 1.0 |
| `-p <float>` | top-p（nucleus sampling），1.0 = 关闭 | 0.9 |
| `-s <int>` | 随机种子 | 当前时间 |
| `-n <int>` | 生成步数，0 = 最大序列长度 | 256 |
| `-i <string>` | 输入 prompt | 空 |
| `-z <string>` | 自定义 tokenizer 路径 | tokenizer.bin |
| `-m <string>` | 模式：generate 或 chat | generate |
| `-y <string>` | chat 模式的 system prompt | 无 |

生成结束后 stderr 会打印 `achieved tok/s`，可用于对比 FP32 与 int8 的速度差异。

## 量化模型

把 FP32 checkpoint 转成 int8（group size 需整除各权重维度，32 对 stories 系列都适用）：

```bash
./quantize stories15M.bin stories15M-q32.bin 32
./runqcpp stories15M-q32.bin -t 0.0 -n 256 -i "Once upon a time"
```

实测（Apple Silicon）：58 MB → 16 MB，tok/s 约 138 → 1343。权重读取量降为 1/4 是 decode 提速的主因——这正是"decode 带宽受限"的直接演示。

注意：int8 量化路径对编译器优化级浮点代码生成（FMA 合并、向量化顺序）非常敏感——1 ulp 的激活差异可能在量化取整时翻转一个 int8，进而让贪心解码在某个接近平局的 argmax 处走向不同的（同样合理的）文本。因此不同编译选项下 runq 的具体生成文本可能不同，这是正常现象；FP32 的 run 不受影响。

## 分步推理教程

[`inference_tutorial/`](inference_tutorial/README_zh.md) 是本仓库配套的动手学习路线，
把两个单文件推理程序拆成 12 个小模块：

1. checkpoint 加载与 BPE 分词
2. RMSNorm、softmax、matmul、RoPE、attention 与 SwiGLU FFN
3. 完整 FP32 forward、采样器与生成循环
4. int8 checkpoint 映射、分组量化、int8 matmul 与量化推理

每个模块都包含概念说明、`main.cpp` 练习模板、`solution.cpp` 参考实现
以及 golden 输入输出数据。可以逐个实现和对比中间结果，最后再组装成
完整模型。程序生成的 `out*.txt` 和本地编译出的模块可执行文件已由
git 忽略，各模块 `data/` 目录中的测试数据仍正常跟踪。

## 调试

macOS 上自带调试器是 **lldb**（没有 gdb）。流程：先用调试参数编译（`-g` 保留符号、`-O0` 关优化，否则变量会被优化掉、断点行号对不上）：

```bash
c++ -g -O0 -std=c++20 -o runcpp_dbg run.cpp
lldb ./runcpp_dbg -- stories15M.bin -t 0.0 -n 8 -i "Once upon a time"
```

lldb 常用命令：

| 命令 | 作用 |
| --- | --- |
| `b main` / `b run.cpp:650` | 在函数名 / 文件行号下断点（行号随代码演进漂移，优先用函数名） |
| `b Transformer::forward` | 在成员函数下断点（建议学推理时断这里） |
| `run`（简写 `r`） | 启动程序 |
| `next`（`n`） | 单步，不进入函数 |
| `step`（`s`） | 单步，进入函数 |
| `finish` | 跑完当前函数返回 |
| `continue`（`c`） | 继续跑到下一个断点 |
| `p x` / `p config.dim` | 打印变量 / 成员 |
| `p s.x` / `p s.q` | 查看整个缓冲区（缓冲区是 `std::vector`，lldb 直接按数组展示内容） |
| `bt` | 查看调用栈 |
| `watch set var pos` | 变量变化时自动停下 |
| `quit`（`q`） | 退出 |

学习推理的推荐断点组合：

```lldb
b Transformer::forward        # 每生成一个 token 停一次，逐层看 x/q/k/v 怎么变
b Sampler::sample             # 看 logits 怎么变成下一个 token
b Tokenizer::encode           # 看 prompt 怎么变成 token 序列
```

在 `forward` 里可以用 `p pos` 看当前位置、`p state.x` 看整个激活缓冲区（lldb 会把 `std::vector` 按数组展示），配合 `finish` 逐层观察——比干读代码理解快得多。

图形化替代：VSCode 装 **CodeLLDB** 扩展，左侧加断点、F5 启动，效果和 lldb 相同但有界面。
