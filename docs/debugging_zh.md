# 调试 <span style="float: right;"><a href="debugging.md">English</a></span>

**macOS** 上自带调试器是 **lldb**（没有 gdb）；**Linux** 上用 **gdb**（下表是 lldb 命令，gdb 的 `break`/`next`/`step`/`finish`/`continue`/`print`/`bt` 用法相同）。流程：先用调试参数编译（`-g` 保留符号、`-O0` 关优化，否则变量会被优化掉、断点行号对不上）：

```bash
# macOS
c++ -g -O0 -std=c++20 -o cpu/runcpp_dbg cpu/run.cpp
lldb ./cpu/runcpp_dbg -- models/stories15M.bin -t 0.0 -n 8 -i "Once upon a time"

# Linux（GCC 11+，例如 g++-12）
g++-12 -g -O0 -std=c++20 -o cpu/runcpp_dbg cpu/run.cpp
gdb --args ./cpu/runcpp_dbg models/stories15M.bin -t 0.0 -n 8 -i "Once upon a time"
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

图形化替代：在 VSCode 里左侧加断点、按 F5 启动。仓库的 `.vscode/launch.json` 按平台提供了两套配置：`Mac: *`（CodeLLDB 扩展，lldb）和 `Linux: *`（C/C++ 扩展，gdb）；各自通过 `preLaunchTask` 先构建对应的 `cpu/*_dbg` 调试二进制再启动。

用 VSCode 时，IntelliSense 与构建任务的标准版本在 `.vscode/c_cpp_properties.json` 和 `.vscode/tasks.json` 里单独配置（当前已设为 `-std=c++20`）；如果改了标准，两边要一起改，否则编辑器会误报找不到 `std::span` 等错误。
