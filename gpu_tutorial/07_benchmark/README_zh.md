# 模块 07：跑分与优化路线 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

## 任务

跑通后第一件事是算理论上限，建立直觉：decode 每步要读完全部权重，所以

```
理论 tok/s ≈ 显存带宽 / 权重字节数
RTX 4080: ≈ 717 GB/s / 60 MB (15M FP32) ≈ 12000 tok/s
```

对比你的实测值（程序结束 stderr 的 `achieved tok/s`），并和 CPU 版（FP32 约 155 tok/s，int8 约 460 tok/s）放在同一表里。

然后用工具看时间花在哪：

```bash
nsys profile -o /tmp/rungpu_prof ./gpu/rungpu models/stories15M.bin -n 256 -i "Once upon a time"
nsys stats /tmp/rungpu_prof.nsys-rep   # 看 kernel 时间占比和启动次数
ncu --set basic ./gpu/rungpu ...        # 看单个 kernel 的带宽利用率（需要权限）
```

实测会低于理论值不少——decode 每步有几十个 kernel 启动，每个 GEMV 又太小打不满带宽。**能解释差距来自哪里，这个模块就算通过**。

## 优化菜单（按性价比排序，每一项都是独立课题）

1. **减少 kernel 启动次数**：rmsnorm+matmul 融合、把三次 GEMV（q/k/v）合并成一次（拼接权重矩阵）、swiglu 融进 GEMV 之后
2. **prefill 用 GEMM**：prompt 阶段的 token 可以批量过 forward，`cublasSgemm` 代替逐 token GEMV，首 token 延迟大幅下降
3. **FP16/BF16 权重**：权重读取减半，decode 直接快一倍（对应 cpu_tutorial 模块 12"带宽受限"的同一个结论）
4. **cuBLASLt epilogue**：matmul + silu + 逐元素乘一次调用完成
5. **flash attention 式融合**：把模块 04 的三阶段合成一个 kernel，省去 att 全局缓冲区的读写

## 验收

一份简短的跑分报告：理论上限、实测 tok/s（GPU vs CPU FP32 vs CPU int8）、nsys 里耗时前三的 kernel，以及你打算先做哪个优化、为什么。
