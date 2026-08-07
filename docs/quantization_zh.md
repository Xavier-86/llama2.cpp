# 量化模型 <span style="float: right;"><a href="quantization.md">English</a></span>

把 FP32 checkpoint 转成 int8（group size 需整除各权重维度，32 对 stories 系列都适用）：

```bash
./cpu/quantize models/stories15M.bin models/stories15M-q32.bin 32
./cpu/runqcpp models/stories15M-q32.bin -t 0.0 -n 256 -i "Once upon a time"
```

实测（Apple Silicon）：58 MB → 16 MB，tok/s 约 138 → 1343。权重读取量降为 1/4 是 decode 提速的主因——这正是"decode 带宽受限"的直接演示。

注意：int8 量化路径对编译器优化级浮点代码生成（FMA 合并、向量化顺序）非常敏感——1 ulp 的激活差异可能在量化取整时翻转一个 int8，进而让贪心解码在某个接近平局的 argmax 处走向不同的（同样合理的）文本。因此不同编译选项下 runq 的具体生成文本可能不同，这是正常现象；FP32 的 run 不受影响。
