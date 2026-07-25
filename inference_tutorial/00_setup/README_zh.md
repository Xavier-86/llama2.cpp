# 00 环境准备：构建参考实现并建立基线 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md)

> 目标：编译参考实现并运行一次，保留这个“正确答案”，后续所有模块都将逐步逼近它。

## 任务

```bash
cd ../..   # 仓库根目录（llama2_cpp）
c++ -O3 -std=c++20 -o runcpp run.cpp
c++ -O3 -std=c++20 -o runqcpp runq.cpp
./runcpp stories15M.bin -t 0.0 -n 64 -s 42 -i "Once upon a time"
./runqcpp stories15M-q32.bin -t 0.0 -n 64 -s 42 -i "Once upon a time"
```

## 验证

第一条命令的输出必须与 `data/expected_greedy.txt` 完全一致（temperature=0 表示贪心解码，因此结果是确定的）：

```bash
diff <(./runcpp stories15M.bin -t 0.0 -n 64 -s 42 -i "Once upon a time" 2>/dev/null) \
     inference_tutorial/00_setup/data/expected_greedy.txt
```

本模块中的 `main.cpp` / `solution.cpp` 只做一个最小健全性检查：打开模型文件并输出其大小。

## 需要观察的两件事（整个项目背后的物理直觉）

1. 查看输出的 `achieved tok/s`。15M 模型在 CPU 上可达到约 100+ tok/s。
2. 对同一提示词，FP32 与 int8 版本是否输出相同的故事？思考量化为何能（或不能）保持相同的贪心路径。

## 背景知识（暂时记住即可，后续模块会逐项展开）

- 本程序只做**前向推理**：加载权重，逐个生成 token，不进行训练。
- 每次调用 `forward(token, pos)` 会处理一个 token，并输出词表上 32000 个分数（logits）；采样器再选择下一个 token。
- 在 decode 阶段，每一步都会读取全部权重，因此速度约等于内存带宽 ÷ 每步读取的字节数——这正是量化（模块 11）能够提速的原因。
