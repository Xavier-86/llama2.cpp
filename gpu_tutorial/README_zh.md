# gpu_tutorial：cuBLAS + 手写 kernel 的 GPU 推理分步教程 <span style="float: right;"><a href="README.md">English</a></span>

[← 项目首页](../README_zh.md)

> 目标：把 `cpu/run.cpp` 的 FP32 推理移植到 GPU——matmul 交给 cuBLAS，其余算子（RMSNorm、RoPE、softmax、attention、SwiGLU）手写 CUDA kernel，最终产物是单文件 `gpu/rungpu.cu`。
>
> 验收标准：同样的 `-t 0.0 -s 42` 贪心解码，输出文本与 `./cpu/runcpp` **逐字一致**。

## 与 cpu_tutorial 的关系

[`cpu_tutorial/`](../cpu_tutorial/README_zh.md) 教你推理本身的原理（每个算子在算什么）；本教程假设你已经读懂 `cpu/run.cpp` 的 `Transformer::forward`（或完成 cpu_tutorial 模块 09），专注另一件事：**把这些算子正确地搬上 GPU**。算子语义不再重复讲解，一律以 `cpu/run.cpp` 为真值来源。

## 环境准备

```bash
sudo apt install nvidia-cuda-toolkit   # 提供 nvcc 和 cuBLAS
nvcc --version                          # 确认装上了
```

编译命令（全程只用这一条，`-arch` 按 GPU 改：RTX 40 系是 `sm_89`，不填则 nvcc 自动检测）：

```bash
nvcc -O3 -std=c++20 -arch=sm_89 -o gpu/rungpu gpu/rungpu.cu -lcublas
```

## 总体设计

### Host / Device 分工

| 留在 CPU（从 `cpu/run.cpp` 原样复制） | 搬到 GPU（本教程的工作） |
| --- | --- |
| checkpoint 加载（`MappedFile`） | 权重的 cudaMemcpy 上传 |
| BPE tokenizer | forward 的全部算子 |
| Sampler（argmax / top-p 采样） | KV cache（常驻显存） |
| generate / chat 循环 | |

### 内存策略

- **权重**：加载后一次性 `cudaMalloc` + `cudaMemcpy` 到显存，之后永不动（stories15M 仅 60 MB，42M 约 210 MB，16 GB 显存无压力）
- **激活缓冲区**（x / xb / xb2 / q / hb / hb2 / att）：显存常驻，对应 `RunState` 的每个成员各一块
- **KV cache**：显存常驻（`n_layers × seq_len × kv_dim × 2 × 4` 字节，15M 模型约 6 MB）
- **logits**：每步把 32000 个 float（128 KB）拷回 CPU 采样——这是每步唯一的回传，代价可忽略

### decode 一个 token 的数据流（每层）

```
x ──rmsnorm──▶ xb ──GEMV──▶ q ──┐
                   ├──GEMV──▶ k ──rope──▶ 写入 KV cache
                   └──GEMV──▶ v ────────▶ 写入 KV cache
q + KV cache ──attention kernel──▶ xb ──GEMV(wo)──▶ xb2 ──残差加──▶ x
x ──rmsnorm──▶ xb ──GEMV(w1)──▶ hb ──┐
                   └──GEMV(w3)──▶ hb2 ──swiglu kernel──▶ hb ──GEMV(w2)──▶ xb ──残差加──▶ x
```

层循环结束后：final rmsnorm → GEMV（分类头）→ logits 拷回 CPU。

## 模块列表

| 模块 | 内容 | 验收 |
| --- | --- | --- |
| [00_setup](00_setup/README_zh.md) | 工程骨架：错误检查宏、权重上传、cuBLAS 句柄 | 编译通过，上传字节数与 checkpoint 对账一致 |
| [01_cublas_matmul](01_cublas_matmul/README_zh.md) | cuBLAS GEMV：row-major 的转置技巧 | 随机输入与 CPU matmul 误差 < 1e-4 |
| [02_rmsnorm](02_rmsnorm/README_zh.md) | block 归约 kernel | 与 CPU 版误差 < 1e-5 |
| [03_rope](03_rope/README_zh.md) | 逐对旋转 kernel | 与 CPU 版误差 < 1e-5 |
| [04_attention](04_attention/README_zh.md) | attention kernel + KV cache（GQA 边界） | 与 CPU 版误差 < 1e-4 |
| [05_elementwise](05_elementwise/README_zh.md) | SwiGLU / 残差加 / embedding 查表 | 与 CPU 版逐 bit 或 1e-6 级一致 |
| [06_forward](06_forward/README_zh.md) | 总装 forward + 与 CPU 逐字对齐 | `diff` 输出为空 |
| [07_benchmark](07_benchmark/README_zh.md) | 跑分、带宽上限、优化路线 | 报出 tok/s 并解释与理论上限的差距 |

学习方法与 cpu_tutorial 相同：**每写一个 kernel 就和 CPU 版的对应中间结果对比，不要等全写完再调**。每个模块三个代码文件：`main.cu`（练习模板，harness 完整、kernel 是 TODO stub）、`solution.cu`（参考实现）、`cases.h`（测试用例 + CPU 参考实现，纯 C++，两处共用）；`data/expected_*.txt` 是 golden 数据，对拍用 `cpu_tutorial/tools/compare.py`。

golden 数据的真值来源是 CPU：`tools/gen_data.cpp`（纯 CPU 程序，包含全部模块的 cases.h）统一生成模块 00–05 的 expected 文件，改用例后重新生成：

```bash
c++ -O2 -std=c++20 -o gpu_tutorial/tools/gen_data gpu_tutorial/tools/gen_data.cpp   # Linux 上 c++ 太旧时用 g++-12
./gpu_tutorial/tools/gen_data gpu_tutorial models/stories15M.bin                    # 仓库根目录运行
```

模块 06 的 `expected_gen.txt` 是 `./cpu/runcpp` 的贪心输出文本（见该模块 README）。模块 07 是跑分与优化，纯文档，无代码。

## 常见坑清单（先扫一眼，踩到了再回来查）

| 症状 | 大概率原因 |
| --- | --- |
| matmul 结果整体错位/乱码 | `lda` 填错，或 OP_T/OP_N 用反（见模块 01） |
| `cublasSgemv` 报 INVALID_VALUE | alpha/beta 传了设备指针；句柄没 create |
| 结果偶尔对偶尔错 | 忘了 kernel 是异步的，host 读结果前没同步（`cudaMemcpy` 自带同步，直接读设备指针才会踩到） |
| 文本在固定位置分叉 | RoPE 的 `kvd` 边界、attention 的 `kv_mul`、softmax 区间 `[0, pos]` 这三处边界条件之一错了（见模块 03/04） |
| 数值差 1e-2 以上 | 不是噪声，是 bug：逐算子 dump 对比定位（见模块 06） |
| 越界/非法地址 | 权重 slice 的偏移算错——所有偏移（`l*dim*dim` 等）直接照抄 CPU 版，一个都别自己重算 |

## 参考

- `cpu/run.cpp` —— 唯一的真值来源，所有算子逻辑以它为准
- `cpu_tutorial/` —— CPU 版分步教程，golden 数据可直接拿来对比中间结果
- [karpathy/llm.c](https://github.com/karpathy/llm.c) —— CUDA kernel 写法参考（GPT-2，但算子同源）
- [cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/) —— `cublasSgemv`/`cublasSgemm` 的参数细节
