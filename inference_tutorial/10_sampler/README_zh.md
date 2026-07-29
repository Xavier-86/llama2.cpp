# 10 sampler：从 logits 选择下一个 token <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

> 目标：实现三种采样策略（贪心 argmax、全分布采样、top-p 核采样）和一个必须逐 bit 复现参考实现的 xorshift 随机数生成器。输入 32000 个分数，输出一个 token id。

## 背景

采样是生成循环的最后一步。每生成一个 token，循环都走一遍：

```
已有 tokens → forward（embedding → 6 层 Transformer → 最终 RMSNorm → 与 wcls 做 matmul）
            → logits[vocab_size] → sampler.sample(logits) → 下一个 token id → 追加，重复
```

模块 09 负责算出 logits；本模块负责"给定 logits，选哪一个 token"。策略由两个超参数决定：`temperature`（温度）和 `topp`（核采样阈值）。

与前面的模块不同，本模块的输出是**离散的 token id**——测试结果要么全对要么错，没有"差不多"。一次浮点尾差就可能让随机数落在 CDF 边界的另一侧，选出另一个 token。所以 RNG 必须逐 bit 一致，所有中间运算也必须用 `float`、按参考实现的顺序做。

## 数学原理

### 策略分派（`sample()`）

```
temperature == 0:   贪心：直接 argmax(logits)，确定性结果，不用随机数
temperature  != 0:  logits_i /= temperature      # 温度缩放，原地修改 logits
                    p = softmax(logits)          # 原地，得到概率分布
                    coin = random_f32()          # 掷一次 [0,1) 的硬币
    topp <= 0 或 >= 1:  全分布采样 sample_mult(p, coin)
    否则:               top-p 核采样 sample_topp(p, topp, coin)
```

温度缩放改变分布的尖锐程度：`temperature < 1` 让高概率 token 更突出（接近贪心），`temperature > 1` 让分布更平（更随机）。注意贪心路径**不做**温度缩放和 softmax——argmax 对单调变换不变。

### 全分布采样（`sample_mult`）

从左到右累加概率（CDF 游走），返回第一个使累计和超过 coin 的下标：

```
cdf = 0
for i in 0..n-1:
    cdf += p_i
    if coin < cdf: return i
return n - 1   # 浮点舍入兜底
```

### top-p 核采样（`sample_topp`）

1. **过滤**：丢弃概率低于 `cutoff = (1 - topp) / (n - 1)` 的 token，幸存者存入 `(prob, index)` 对；
2. **排序**：幸存者按概率**降序**排序；
3. **截断**：从头累加，保留使累计概率**首次超过** `topp` 的最小前缀（若一直不超过则保留全部），记下该前缀的总概率 `cumulative_prob`；
4. **重归一化并掷点**：等效于把前缀内的概率除以 `cumulative_prob` 再按 CDF 选——实现上用 `r = coin * cumulative_prob` 在前缀内做一次 CDF 游走即可，不必真的逐元素除。

### xorshift 随机数生成器（必须与参考实现逐 bit 一致）

64 位状态 `st`（种子即初始状态），每次取数：

```
st ^= st >> 12
st ^= st << 25
st ^= st >> 27
u32 = (st * 0x2545F4914F6CDD1D) >> 32    # 64 位无符号乘，取高 32 位
```

映射到 `[0, 1)` 的 float 硬币（丢弃低 8 位，剩 24 位正好是 float 尾数精度）：

```
coin = (u32 >> 8) / 16777216.0f          # 16777216 = 2^24
```

`data/expected_rng_seed42.txt` 保存 seed=42 时的前 10 个硬币——先单独验证 RNG，再测采样。

## 输入数据

| 变量 | 位置 | 形状 | 布局 | 含义 | 来自模型哪里 |
| --- | --- | --- | --- | --- | --- |
| `kLogitsSynth` | main.cpp（const 数组） | (8,) | float 一维 | 合成 logits `[1..8]`，平坦斜坡让每个 token 概率明显不同 | 无（人造数据，专门覆盖 mult / top-p 路径） |
| `real` | main() 内局部变量，由 `tut::read_floats("data/input_logits.txt")` 加载 | (32000,) | float 一维，每行一个值 | 真实 logits 向量 | 参考提示词 "Once upon a time"（token id `[1, 9038, 2501, 263, 931]`，P=5）**最后一个提示位置**（pos=4）经模块 09 完整前向后的输出：最终 RMSNorm 与 `wcls` matmul 的结果 |

**为什么真实 logits 还从文件读？** 这是全教程唯一的例外：32000 个浮点值内嵌成 const 数组会让源文件膨胀到约 400 KB，得不偿失，因此保留数据文件，用一行 `tut::read_floats` 加载（声明在 `../common/io.h`）。本模块没有 data.h。

模型常量：`vocab_size = 32000`（真实用例的 logits 长度）；合成用例 vocab = 8。

## 任务

补全 `main.cpp` 中的 `TODO`（都在 `Sampler` 类和 `softmax` 里）：

1. **task 1 — `random_u32()` / `random_f32()`**：按上面的公式推进 xorshift 状态并取高 32 位（注意是 64 位无符号乘），再把 u32 映射成 `[0, 1)` 的 float。先让 out_rng.txt 对上再继续。
2. **task 2 — `softmax(x)`**：原地、数值稳定（先减最大值再 exp）。
3. **task 3 — `sample(logits)`**：按"策略分派"一节分派；**task 3a — `sample_argmax`**：返回最大元素下标。
4. **task 4 — `sample_mult(probabilities, coin)`**：CDF 游走。
5. **task 5 — `sample_topp(probabilities, topp, coin)`**：过滤 → 降序排序 → 截断 → 用 `coin * cumulative_prob` 在前缀内掷点。

`main()` 已经写好：先做 RNG 自检写 `out_rng.txt`，再跑 8 个用例写 `out.txt`。

## 测试用例

前 4 个用真实 logits，后 4 个用合成 logits，参数为 (temperature, topp, seed)：

```
1. (0.0, 0.9, 42)   real  -> 贪心
2. (1.0, 1.0, 42)   real  -> 全分布采样
3. (0.8, 0.9, 42)   real  -> top-p
4. (0.8, 0.9, 1234) real  -> top-p，换 seed
5. (0.0, 0.9, 42)   synth -> 贪心，预期 7
6. (1.0, 1.0, 42)   synth -> 全分布采样，预期 6
7. (1.0, 0.5, 42)   synth -> top-p，预期 7
8. (2.0, 0.9, 7)    synth -> top-p，预期 5
```

真实分布有 96.6% 的概率集中在 argmax（token 29892）上，所以 4 个真实用例都选中它——真正考验 mult / top-p 实现的是合成用例。

## 构建 / 运行 / 验证

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out_rng.txt data/expected_rng_seed42.txt
python3 ../tools/compare.py out.txt data/expected_samples.txt --exact
```

`out.txt` 是 token id，必须用 `--exact` 精确比较。

## 常见错误

- **`sample()` 会原地修改输入 logits**（先除以 temperature 再 softmax）。这是参考实现的约定——真实推理里 logits 缓冲区每个位置都会重算，无需保留。测试驱动为每个用例复制了一份，你在 `sample` 内部不要试图"保护"输入。
- RNG 对不上 → 检查：移位量（12 / 25 / 27）、乘法是否 64 位无符号、是否取**高** 32 位、`>> 8` 和除以 `16777216.0f` 是否照抄。
- top-p 全错 → 排序方向必须是概率**降序**；`cutoff = (1 - topp) / (n - 1)` 里的 `n` 是完整词表大小（32000 或 8），不是幸存者数量。
- 只有一个用例失败 → 多半是浮点尾差让 coin 落在 CDF 边界另一侧：检查是否哪里用了 `double`、softmax 是否减了最大值、温度缩放是否在 softmax **之前**。
- 贪心路径做了 softmax/掷硬币 → 不仅浪费，还会多消耗一个随机数（本模块每个用例各自新建 Sampler，不会暴露，但真实生成循环会错位）。

## 完成标准

两项比较全部 PASS（RNG 浮点比较 + 8 个 token id 精确比较）。`solution.cpp` 是参考答案——卡住再看，看完关掉自己写。
