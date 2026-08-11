#!/usr/bin/env bash
set -euo pipefail

gpu_nvcc=${NVCC:-nvcc}
host_cxx=${HOST_CXX:-g++-13}
bench_dir=${TMPDIR:-/tmp}/llama2-4080s-bench
mkdir -p "$bench_dir"

common_flags=(-O3 -std=c++20 -ccbin "$host_cxx" -arch=sm_89)
"$gpu_nvcc" "${common_flags[@]}" -o "$bench_dir/rungpu" gpu/default/rungpu.cu -lcublas
"$gpu_nvcc" "${common_flags[@]}" -o "$bench_dir/runqgpu-default" gpu/default/runqgpu.cu
"$gpu_nvcc" "${common_flags[@]}" -o "$bench_dir/runqgpu-4080s" gpu/4080s/runqgpu.cu

echo "== end-to-end decode (three runs per variant) =="
for model in stories15M stories42M; do
    for variant in fp32 int8 4080s; do
        if [[ $variant == fp32 ]]; then
            binary="$bench_dir/rungpu"
            checkpoint="models/$model.bin"
        elif [[ $variant == int8 ]]; then
            binary="$bench_dir/runqgpu-default"
            checkpoint="models/$model-q32.bin"
        else
            binary="$bench_dir/runqgpu-4080s"
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
