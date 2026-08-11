#!/usr/bin/env bash
set -euo pipefail

ppu_nvcc=${NVCC:-nvcc}
bench_dir=${TMPDIR:-/tmp}/llama2-ppu-bench
mkdir -p "$bench_dir"

"$ppu_nvcc" -O3 -std=c++20 -o "$bench_dir/rungpu" gpu/default/rungpu.cu -lcublas
"$ppu_nvcc" -O3 -std=c++20 -o "$bench_dir/runqgpu-default" gpu/default/runqgpu.cu
"$ppu_nvcc" -O3 -std=c++20 -o "$bench_dir/runqgpu-ppu" gpu/ppu/runqgpu.cu
"$ppu_nvcc" -O3 -std=c++20 -o "$bench_dir/test_ppu_qgemv" gpu/ppu/test_qgemv.cu

echo "== kernel correctness and QKV microbenchmark =="
"$bench_dir/test_ppu_qgemv" 500

echo "== end-to-end decode (three runs per variant) =="
for model in stories15M stories42M; do
    for variant in fp32 int8 ppu; do
        if [[ $variant == fp32 ]]; then
            binary="$bench_dir/rungpu"
            checkpoint="models/$model.bin"
        elif [[ $variant == int8 ]]; then
            binary="$bench_dir/runqgpu-default"
            checkpoint="models/$model-q32.bin"
        else
            binary="$bench_dir/runqgpu-ppu"
            checkpoint="models/$model-q32.bin"
        fi
        printf '%-16s %-8s' "$model" "$variant"
        for run in 1 2 3; do
            output=$({ "$binary" "$checkpoint" \
                -t 0.0 -n 256 -s 42 -i "Once upon a time" >/dev/null; } 2>&1)
            rate=$(sed -n 's/^achieved tok\/s: //p' <<<"$output")
            printf '  %s tok/s' "$rate"
        done
        printf '\n'
    done
done
