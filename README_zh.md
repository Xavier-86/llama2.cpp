# llama2.cpp：极简 CPU 与 GPU LLM 推理 <span style="float: right;"><a href="README.md">English</a></span>

> 一个面向学习的 Llama 2 推理实现，基于 [karpathy/llama2.c](https://github.com/karpathy/llama2.c) 开发。
> 项目包含 CPU 与 GPU 两套实现：CPU 版本使用现代 C++20 编写，支持 FP32 和 int8 推理；GPU 版本基于 CUDA，通过 cuBLAS 与手写 kernel 实现完整推理流程。两套实现采用统一的模型格式和命令行接口，方便对照阅读源码、验证数值一致性，并比较两者的推理性能。

## 项目结构

```
├── cpu/                  CPU 推理：run.cpp（FP32）、runq.cpp（int8）、quantize.cpp（转换工具）
├── gpu/                  GPU 推理：预设、RTX 4080s 与 PPU 三套实现
├── models/               权重与分词器：stories15M/42M（.bin，FP32）、stories*-q32.bin（int8）、tokenizer.bin
├── cpu_tutorial/         CPU 分步教程：用 12 个可验证模块亲手重建完整推理流程
├── gpu_tutorial/         GPU 分步教程：用 8 个模块把 forward 移植到 CUDA
└── docs/                 详细文档：用法、量化、调试
```

## 环境要求

- **支持 C++20 的编译器**（`std::span`、`std::ranges`、`std::from_chars`）：
  - macOS：系统自带 `/usr/bin/clang++`（Apple Clang 15+，装 Xcode Command Line Tools 即可）
  - Linux：GCC 11+ 或 Clang 14+，例如 `sudo apt install g++-12`。如果默认 `c++` 太旧（Ubuntu 的 GCC 9.5 连 `-std=c++20` 都不认），用带版本号的命令：`g++-12 -O3 -std=c++20 ...`
- **模型权重**：从 [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) 下载 `stories15M.bin` / `stories42M.bin` 并放入 `models/`。权重文件由 Git LFS 管理，请确保已拉取完整文件。仓库已包含 `models/tokenizer.bin`。

## 快速开始

```bash
# 编译（Linux 上如有必要把 c++ 换成 g++-12，见上）
c++ -O3 -std=c++20 -o cpu/runcpp cpu/run.cpp        # FP32 推理
c++ -O3 -std=c++20 -o cpu/runqcpp cpu/runq.cpp      # int8 量化推理
c++ -O3 -std=c++17 -o cpu/quantize cpu/quantize.cpp # 量化转换工具

# 运行
./cpu/runcpp models/stories15M.bin -t 0.0 -n 256 -s 42 -i "Once upon a time"
./cpu/runqcpp models/stories15M-q32.bin -t 0.0 -n 256 -s 42 -i "Once upon a time"

# 自己生成 int8 checkpoint
./cpu/quantize models/stories15M.bin models/stories15M-q32.bin 32
```

生成结束后 stderr 会打印 `achieved tok/s`，可用于对比 FP32 与 int8 的速度差异。

GPU 编译方法及 CPU/GPU 性能对比见 [gpu/README_zh.md](gpu/README_zh.md)。GPU 代码包含预设实现，以及分别面向 RTX 4080 SUPER 和真武 810E PPU 的 int8 kernel。

## 文档

- [docs/usage_zh.md](docs/usage_zh.md) — 全部命令行参数、generate/chat 模式、示例
- [docs/quantization_zh.md](docs/quantization_zh.md) — int8 格式、group size、速度与体积权衡
- [docs/debugging_zh.md](docs/debugging_zh.md) — lldb（macOS）/ gdb（Linux）/ VSCode 调试指南
- [cpu_tutorial/](cpu_tutorial/README_zh.md) — 动手教程：借助 golden 测试数据逐模块重实现 FP32 与 int8 推理
- [gpu_tutorial/](gpu_tutorial/README_zh.md) — GPU 移植教程：cuBLAS + 手写 CUDA kernel，8 个模块把 forward 搬上显卡
