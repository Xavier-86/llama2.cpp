# GPU 推理 <span style="float: right;"><a href="README.md">English</a></span>

GPU 实现按目标硬件拆成三个目录。FP32 路径由 cuBLAS 执行，int8 路径使用手写 CUDA kernel；checkpoint 加载、tokenizer、sampler 和 CLI 与 CPU 版本共用。

| 目录 | 内容 | 用途 |
| --- | --- | --- |
| `default/` | `rungpu.cu`、`runqgpu.cu` | 预设 FP32、普通 int8 与共享实现 |
| `4080s/` | `runqgpu.cu`、`bench.sh` | RTX 4080 SUPER 专用融合 int8 GEMV 与测试 |
| `ppu/` | `runqgpu.cu`、`test_qgemv.cu`、`bench.sh` | 真武 810E PPU 专用 int8 GEMV 与测试 |

`default/runqgpu_impl.cuh` 是三种 int8 入口共用的内部实现。每个 `runqgpu.cu` 在编译期固定自己的 kernel，不需要也不接受 `-k` 参数。

## RTX 4080 SUPER

```bash
# CUDA 12.8，源码已在 sm_89 上验证
nvcc -O3 -std=c++20 -ccbin g++-13 -arch=sm_89 \
  -o /tmp/rungpu gpu/default/rungpu.cu -lcublas
nvcc -O3 -std=c++20 -ccbin g++-13 -arch=sm_89 \
  -o /tmp/runqgpu-default gpu/default/runqgpu.cu
nvcc -O3 -std=c++20 -ccbin g++-13 -arch=sm_89 \
  -o /tmp/runqgpu-4080s gpu/4080s/runqgpu.cu

/tmp/runqgpu-4080s models/stories15M-q32.bin \
  -t 0.0 -n 256 -s 42 -i "Once upon a time"
```

实测参数：`-t 0.0 -n 256 -s 42 -i "Once upon a time"`，量化 group size 为 32。

| 模型 | FP32 | int8 | RTX 4080s 优化 int8 |
| --- | ---: | ---: | ---: |
| stories15M | ~2660 tok/s | ~2200 tok/s | ~3060 tok/s |
| stories42M | ~1490 tok/s | ~1510 tok/s | ~2100 tok/s |

4080s 版本使用融合的 warp-per-row GEMV：`float4`/`int8x4` 向量化加载、分段 warp 归约、在线激活量化和 `__dp4a` 点积，避免单独启动量化 kernel。

三种实现各跑三轮：`./gpu/4080s/bench.sh`。

## 真武 810E PPU

PPU SDK v2.0 使用 CUDA 兼容工具链。不要传 `-arch=sm_89`，让编译器生成 PPU 原生目标；如果仓库位于 ossfs，建议把可执行文件输出到 `/tmp`。

```bash
nvcc -O3 -std=c++20 -o /tmp/rungpu gpu/default/rungpu.cu -lcublas
nvcc -O3 -std=c++20 -o /tmp/runqgpu-default gpu/default/runqgpu.cu
nvcc -O3 -std=c++20 -o /tmp/runqgpu-ppu gpu/ppu/runqgpu.cu

/tmp/runqgpu-ppu models/stories15M-q32.bin \
  -t 0.0 -n 256 -s 42 -i "Once upon a time"
```

| 模型 | FP32 | int8 | PPU 优化 int8 |
| --- | ---: | ---: | ---: |
| stories15M | ~1660 tok/s | ~1640 tok/s | ~2317 tok/s |
| stories42M | ~1136 tok/s | ~944 tok/s | ~1078 tok/s |

PPU 版本针对原融合 kernel 在 810E 上的低效点重新设计：每个 warp 并行处理四个 32 元素量化组，激活只量化一次并复用，Q/K/V 合并为一次 launch，W1/W3 合并为一次 launch，GEMV 使用 `int8x4 + __dp4a`。RMSNorm 和 SwiGLU 会直接生成量化激活，不再把中间 FP32 向量写回后另起量化 kernel；每层由此再减少 3 次 launch。当前要求 checkpoint 使用 `GS=32`，其他 group size 会明确报错。

融合前后在同一台 810E、相同编译参数及上述 decode 参数下各跑三轮，平均结果如下：

| 模型 | 融合前 | RMSNorm/SwiGLU 融合后 | 提升 |
| --- | ---: | ---: | ---: |
| stories15M | 2291 tok/s | 2330 tok/s | 1.7% |
| stories42M | 1244 tok/s | 1283 tok/s | 3.1% |

完整测试会先逐元素对比 CPU 参考结果，再运行 QKV microbenchmark，最后对两个模型的 FP32、普通 int8 和 PPU 优化 int8 各测试三次：

```bash
./gpu/ppu/bench.sh
```

早期版本在真武 810E（驱动 1.3.2-d7f5a2）上的 PPU 优化 int8 三轮结果为 `2316.67 / 2316.67 / 2316.67 tok/s`（stories15M）和 `1079.30 / 1047.01 / 1108.60 tok/s`（stories42M）；顶部对比表沿用这组历史数据。kernel 正确性测试中，Q/K/V/W1/W3 投影相对 CPU 参考的最大绝对误差为 `4.76837e-06`；融合的 RMSNorm/量化与 SwiGLU/量化也会和原两步实现逐元素对拍。QKV microbenchmark 从 `18.53 us` 降至 `3.95 us`，加速 `4.69x`。

## 分步教程

[`gpu_tutorial/`](../gpu_tutorial/README_zh.md) 用八个模块讲解从权重上传到完整 CUDA forward；其中 cuBLAS FP32 路径对应这里的 `default/rungpu.cu`。
