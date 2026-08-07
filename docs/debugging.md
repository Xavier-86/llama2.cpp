# Debugging <span style="float: right;"><a href="debugging_zh.md">中文</a></span>

**macOS**: the bundled debugger is **lldb** (there is no gdb). **Linux**: use **gdb** (the commands below are lldb's; gdb's `break`/`next`/`step`/`finish`/`continue`/`print`/`bt` work the same way). Workflow: first build with debug flags (`-g` keeps symbols, `-O0` disables optimization — otherwise variables get optimized away and breakpoint line numbers won't match):

```bash
# macOS
c++ -g -O0 -std=c++20 -o cpu/runcpp_dbg cpu/run.cpp
lldb ./cpu/runcpp_dbg -- models/stories15M.bin -t 0.0 -n 8 -i "Once upon a time"

# Linux (GCC 11+; e.g. g++-12)
g++-12 -g -O0 -std=c++20 -o cpu/runcpp_dbg cpu/run.cpp
gdb --args ./cpu/runcpp_dbg models/stories15M.bin -t 0.0 -n 8 -i "Once upon a time"
```

Common lldb commands:

| Command | Effect |
| --- | --- |
| `b main` / `b run.cpp:650` | Break at a function name / file line (line numbers drift as code evolves; prefer function names) |
| `b Transformer::forward` | Break at a member function (recommended when learning inference) |
| `run` (`r`) | Start the program |
| `next` (`n`) | Step over |
| `step` (`s`) | Step into |
| `finish` | Run to the end of the current function |
| `continue` (`c`) | Continue to the next breakpoint |
| `p x` / `p config.dim` | Print a variable / member |
| `p s.x` / `p s.q` | Show a whole buffer (buffers are `std::vector`; lldb renders them as arrays) |
| `bt` | Show the call stack |
| `watch set var pos` | Stop automatically when a variable changes |
| `quit` (`q`) | Quit |

Recommended breakpoint set for learning inference:

```lldb
b Transformer::forward        # stop once per generated token, watch x/q/k/v change layer by layer
b Sampler::sample             # see how logits become the next token
b Tokenizer::encode           # see how the prompt becomes a token sequence
```

Inside `forward` you can use `p pos` for the current position and `p state.x` for the whole activation buffer (lldb renders `std::vector` as an array), stepping through layers with `finish` — much faster than just reading the code.

GUI alternative: in VSCode, set breakpoints in the gutter and press F5. The repo's `.vscode/launch.json` has per-platform configurations: `Mac: *` (CodeLLDB extension, lldb) and `Linux: *` (C/C++ extension, gdb); each builds the matching `cpu/*_dbg` binary via its `preLaunchTask` before launching.

With VSCode, the language standard is configured separately for IntelliSense and for build tasks in `.vscode/c_cpp_properties.json` and `.vscode/tasks.json` (currently set to `-std=c++20`). If you change the standard, change it in both places — otherwise the editor will falsely report errors like "no member named `span` in namespace `std`".
