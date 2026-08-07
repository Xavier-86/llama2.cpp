// Pure-CPU golden-data generator for gpu_tutorial. Golden data is compared with
// module solution GPU output, but the CPU reference is the source of truth.
// All inputs come from each module's cases.h as the single source of truth;
// after changing a case, rerun this tool.
//
// usage: gen_data <out_root> <checkpoint>
// e.g.:  ./gen_data gpu_tutorial models/stories15M.bin   (run from the repository root)
//
// Note: this tool does not generate 06_forward/data/expected_gen.txt. It is the
//   ./cpu/runcpp models/stories15M.bin -t 0.0 -n 128 -s 42 -i "Once upon a time"
// command's standard output (see 06_forward/README.md).

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../cpu_tutorial/common/io.h"
#include "../00_setup/cases.h"
#include "../01_cublas_matmul/cases.h"
#include "../02_rmsnorm/cases.h"
#include "../03_rope/cases.h"
#include "../04_attention/cases.h"
#include "../05_elementwise/cases.h"

namespace {

std::string out_dir(const std::string& root, const std::string& module) {
    const std::string dir = root + "/" + module + "/data";
    std::filesystem::create_directories(dir);
    return dir + "/";
}

void gen_00(const std::string& root, const std::string& ckpt) {
    std::ifstream f(ckpt, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot open " + ckpt); }
    int cfg[7];
    if (!f.read(reinterpret_cast<char*>(cfg), sizeof(cfg))) {
        throw std::runtime_error("cannot read config header of " + ckpt);
    }
    tut::write_ints(out_dir(root, "00_setup") + "expected_upload.txt", gt::tensor_byte_table(cfg));
}

void gen_01(const std::string& root) {
    const std::string dir = out_dir(root, "01_cublas_matmul");
    {
        float y[gt::kToyD];
        gt::matmul_cpu(y, gt::kToyX, gt::kToyW, gt::kToyN, gt::kToyD);
        tut::write_floats(dir + "expected_toy.txt", y);
    }
    {
        const std::vector<float> w = gt::make_real_w();
        const std::vector<float> x = gt::make_real_x();
        std::vector<float> y(gt::kRealD);
        gt::matmul_cpu(y.data(), x.data(), w.data(), gt::kRealN, gt::kRealD);
        tut::write_floats(dir + "expected_real.txt", y);
    }
}

void gen_02(const std::string& root) {
    const std::string dir = out_dir(root, "02_rmsnorm");
    {
        float o[gt::kToySize];
        gt::rmsnorm_cpu(o, gt::kToyVec, gt::kToyWeight, gt::kToySize);
        tut::write_floats(dir + "expected.txt", o);
    }
    {
        const std::vector<float> x = gt::make_real_vec();
        const std::vector<float> w = gt::make_real_weight();
        std::vector<float> o(gt::kRealSize);
        gt::rmsnorm_cpu(o.data(), x.data(), w.data(), gt::kRealSize);
        tut::write_floats(dir + "expected_real.txt", o);
    }
}

void gen_03(const std::string& root) {
    const std::string dir = out_dir(root, "03_rope");
    {
        std::vector<float> q(std::begin(gt::kToyQ), std::end(gt::kToyQ));
        std::vector<float> k(std::begin(gt::kToyK), std::end(gt::kToyK));
        gt::rope_cpu(q.data(), k.data(), gt::kToyPos, gt::kToyDim, gt::kToyKvd, gt::kToyHeadSize);
        tut::write_floats(dir + "expected_q.txt", q);
        tut::write_floats(dir + "expected_k.txt", k);
    }
    {
        std::vector<float> q = gt::make_real_q();
        std::vector<float> k = gt::make_real_k();
        gt::rope_cpu(q.data(), k.data(), gt::kRealPos, gt::kRealDim, gt::kRealKvd, gt::kRealHeadSize);
        tut::write_floats(dir + "expected_q_real.txt", q);
        tut::write_floats(dir + "expected_k_real.txt", k);
    }
}

void gen_04_case(const std::string& dir, const std::string& tag,
                 const std::vector<float>& q, const std::vector<float>& kc,
                 const std::vector<float>& vc, int pos, int n_heads, int kvd,
                 int kv_mul, int head_size, int seq_len) {
    std::vector<float> xb((size_t)n_heads * head_size);
    std::vector<float> att((size_t)n_heads * (pos + 1));
    gt::attention_cpu(xb.data(), att.data(), q.data(), kc.data(), vc.data(),
                      pos, n_heads, kvd, kv_mul, head_size, seq_len);
    tut::write_floats(dir + "expected_xb_" + tag + ".txt", xb);
    tut::write_floats(dir + "expected_att_" + tag + ".txt", att);
}

void gen_04(const std::string& root) {
    const std::string dir = out_dir(root, "04_attention");
    gen_04_case(dir, "a", gt::make_a_q(), gt::make_a_k(), gt::make_a_v(),
                gt::kAPos, gt::kANHeads, gt::kAKvd, gt::kAKVMul, gt::kAHeadSize, gt::kASeqLen);
    gen_04_case(dir, "b", gt::make_b_q(), gt::make_b_k(), gt::make_b_v(),
                gt::kBPos, gt::kBNHeads, gt::kBKvd, gt::kBKVMul, gt::kBHeadSize, gt::kBSeqLen);
}

void gen_05(const std::string& root) {
    const std::string dir = out_dir(root, "05_elementwise");
    {
        std::vector<float> hb(std::begin(gt::kSwigluHb), std::end(gt::kSwigluHb));
        gt::swiglu_cpu(hb.data(), gt::kSwigluHb2, gt::kSwigluN);
        tut::write_floats(dir + "expected_swiglu.txt", hb);
    }
    {
        std::vector<float> x(std::begin(gt::kAddX), std::end(gt::kAddX));
        gt::add_cpu(x.data(), gt::kAddY, gt::kAddN);
        tut::write_floats(dir + "expected_add.txt", x);
    }
    {
        float x[gt::kEmbedDim];
        gt::embed_cpu(x, gt::kEmbedTable, gt::kEmbedToken, gt::kEmbedDim);
        tut::write_floats(dir + "expected_embed.txt", x);
    }
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: gen_data <out_root> <checkpoint>\n");
        return 1;
    }
    const std::string root = argv[1], ckpt = argv[2];
    gen_00(root, ckpt);
    gen_01(root);
    gen_02(root);
    gen_03(root);
    gen_04(root);
    gen_05(root);
    std::printf("golden data written under %s/*/data/\n", root.c_str());
    std::printf("note: 06_forward/data/expected_gen.txt is produced by cpu/runcpp, "
                "see 06_forward/README_zh.md\n");
    return 0;
}
