# 00 环境准备：构建参考实现并建立基线 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

## 总任务

分两部分：(1) 在仓库根目录编译参考实现并运行一次，保留这个"正确答案"，后续所有模块都将逐步逼近它；(2) 补全 `main.cpp` 中的三个 TODO——一个最小健全性检查：打开后续所有模块都依赖的三个二进制文件并输出其大小。

参考程序只做**前向推理**：

```
权重（stories15M.bin）+ 词表（tokenizer.bin）
        |
        v
forward(token, pos) -> 词表上 32000 个 logits -> 采样器 -> 下一个 token
        ^                                                  |
        |______________ 每步喂回一个 token ________________|
```

不进行训练：加载权重，逐个生成 token。

**输入**：仓库根目录下的三个二进制文件（以下路径相对于本模块目录），无需读任何数据文件：

| 文件 | 使用者 | 含义 |
| --- | --- | --- |
| `../../models/stories15M.bin` | `../../cpu/run.cpp` | FP32 参考模型 |
| `../../models/stories15M-q32.bin` | `../../cpu/runq.cpp` | int8 量化模型，group size 32 |
| `../../models/tokenizer.bin` | 两者 | 词表 / 分词器表 |

**输出**：`./cpu/runcpp ... -t 0.0 -n 64 -s 42 -i "Once upon a time"` 打印的故事必须与 `data/expected_greedy.txt` 完全一致（temperature=0 表示贪心解码，因此结果是确定的）。`data/expected_greedy.txt` 是黄金数据，不要修改。健全性检查的 `main()` 会把上述三个文件的大小逐行打印出来。

## 子任务一：构建参考实现并跑出基线

`main.cpp` 中没有对应的 TODO——这个子任务是仓库根目录下的纯命令行工作：

```bash
cd ../..   # 仓库根目录（llama2_cpp）
c++ -O3 -std=c++20 -o cpu/runcpp cpu/run.cpp
c++ -O3 -std=c++20 -o cpu/runqcpp cpu/runq.cpp
./cpu/runcpp models/stories15M.bin -t 0.0 -n 64 -s 42 -i "Once upon a time"
./cpu/runqcpp models/stories15M-q32.bin -t 0.0 -n 64 -s 42 -i "Once upon a time"
```

然后对黄金文件验证第一条命令的输出（在仓库根目录下执行）：

```bash
diff <(./cpu/runcpp models/stories15M.bin -t 0.0 -n 64 -s 42 -i "Once upon a time" 2>/dev/null) \
     cpu_tutorial/00_setup/data/expected_greedy.txt
```

运行时观察两件事（整个项目背后的物理直觉）：

1. 查看输出的 `achieved tok/s`。15M 模型在 CPU 上可达到约 100+ tok/s。
2. 对同一提示词，FP32 与 int8 版本是否输出相同的故事？思考量化为何能（或不能）保持相同的贪心路径。

需要的知识——前向推理，以及量化为何能提速（暂时记住即可，后续模块会逐项展开）：

- 每次调用 `forward(token, pos)` 会处理一个 token，并输出词表上 32000 个分数（logits）；采样器再选择下一个 token。
- 在 decode 阶段，每一步都会读取全部权重，因此速度约等于内存带宽 ÷ 每步读取的字节数——这正是量化（模块 12）能够提速的原因。

## 子任务二：`print_file_size(path)`——TODO(task 1) 和 TODO(task 2)

`main.cpp` 中的健全性检查（本模块没有真正的计算）。按顺序做两件事：

1. 以二进制模式打开 `path`；若打不开，向 stderr 打印错误信息并返回 false。
2. 求出文件的字节大小，单独一行打印到 stdout，然后返回 true。

需要的知识——这个函数要处理的三个文件就是上方输入表里的三个；`main.cpp` 已经把路径定义为 `kModelFp32`、`kModelInt8` 和 `kTokenizer`。

## 子任务三：串联 `main()`——TODO(task 3)

对三个文件（FP32 模型、int8 模型、分词器）各调用一次 `print_file_size`；仅当三个都成功时返回 0，否则返回 1。

## 构建 / 运行 / 验证

在本模块目录下编译并运行健全性检查：

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
```

参考实现的构建命令和 `diff` 验证命令见子任务一。

## 常见错误

- `diff` 报找不到文件 → 验证命令必须在仓库根目录下执行（黄金文件路径以 `cpu_tutorial/...` 开头）；而 `main.cpp` 里的路径是相对于本模块目录的。
- 打印的故事与 `data/expected_greedy.txt` 不一致 → 检查参数：只有 `-t 0.0`（贪心解码）是确定的，其他 temperature 都是随机采样。

## 完成标准

子任务一的 `diff` 没有任何差异输出，且 `./main` 逐行打印三个文件的大小并以 0 退出。`solution.cpp` 是参考答案——卡住再看，看完关掉自己写。
