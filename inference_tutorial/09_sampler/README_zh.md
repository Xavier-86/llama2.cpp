# 09 sampler：从 logits 选择下一个 token <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

> 目标：实现三种采样策略。输入 32000 个分数，输出一个 token id。

## 三种策略

```
temperature == 0:   直接 argmax（贪心、确定性）
topp <= 0 或 >= 1:  全分布采样：logits /= temperature -> softmax
                    -> 生成随机数并按 CDF 选取
其他情况:           top-p（核采样）：先同上得到概率；仅保留概率
                    >= (1-topp)/(n-1) 的 token；按概率降序排序；取累计概率
                    超过 topp 的最小集合；使用 coin * set_total_prob 在集合内采样
```

## 随机数生成器（必须与参考实现逐 bit 一致）

使用 uint64 状态的 xorshift：

```
state ^= state >> 12;  state ^= state << 25;  state ^= state >> 27
u32 = (state * 0x2545F4914F6CDD1D) >> 32
coin = (u32 >> 8) / 16777216.0f        # [0, 1) 内的 float
```

`data/expected_rng_seed42.txt` 保存 seed=42 时的前 10 个随机数；测试采样前先验证 RNG。

## 数据文件

| 文件 | 内容 |
| --- | --- |
| `input_logits.txt` | 真实 logits（提示词最后位置，32000 个值） |
| `input_logits_synth.txt` | 合成 logits `[1..8]`（vocab=8） |
| `expected_rng_seed42.txt` | seed 42 的前 10 个随机数 |
| `expected_samples.txt` | 下面 8 个用例的预期 token id |

用例顺序——前 4 个使用真实 logits，后 4 个使用合成数据，参数为 (temperature, topp, seed)：

```
1. (0.0, 0.9, 42)   -> greedy
2. (1.0, 1.0, 42)   -> full distribution
3. (0.8, 0.9, 42)   -> top-p
4. (0.8, 0.9, 1234) -> top-p，不同 seed
5. (0.0, 0.9, 42)   synth -> greedy，预期 7
6. (1.0, 1.0, 42)   synth -> full distribution，预期 6
7. (1.0, 0.5, 42)   synth -> top-p，预期 7
8. (2.0, 0.9, 7)    synth -> top-p，预期 5
```

真实分布有 96.6% 的概率集中在 argmax 上，因此真实用例都会选中它；真正覆盖多项分布 / top-p 路径的是合成用例。

## 任务与验证

```bash
python3 ../tools/compare.py out_rng.txt data/expected_rng_seed42.txt
python3 ../tools/compare.py out.txt data/expected_samples.txt --exact
```

## 提示

- **采样会修改输入 logits**：先原地除以 temperature，再做 softmax。每个用例前都要复制一份。
- top-p 必须按概率**降序**排列，使用带自定义比较器的 `std::sort` 即可。
- 浮点尾差可能使随机数恰好落在 CDF 边界上；这些用例特意避开了该问题。若仅一个用例失败，输出中间概率进行比较。
