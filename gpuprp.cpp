// GPU PRP prototype for 2^n - c on OpenCL (zero-padded FP64 FFT).
// usage: gpuprp n [c=25] [w=18] [iters=0 (full test)] [outfile]
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(e)                                                               \
    do {                                                                       \
        cl_int _e = (e);                                                       \
        if (_e != CL_SUCCESS) {                                                \
            std::fprintf(stderr, "OpenCL error %d at line %d\n", _e, __LINE__); \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static const uint32_t BLK = 16;
static uint32_t pad(uint32_t v) { return (v + BLK - 1) / BLK * BLK; }

struct Ctx {
    cl_context ctx;
    cl_command_queue q;
    cl_program prog;
    cl_kernel k_load, k_r4, k_r2, k_sq, k_round, k_cA, k_cB, k_chk, k_fold;
};

static void run1(Ctx& c, cl_kernel k, size_t global) {
    size_t local = 64;
    size_t g = (global + local - 1) / local * local;
    CHECK(clEnqueueNDRangeKernel(c.q, k, 1, nullptr, &g, &local, 0, nullptr, nullptr));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: gpuprp n [c=25] [w=18] [iters=0 full] [outfile]\n");
        return 1;
    }
    const uint32_t n = std::atoi(argv[1]);
    const long long cc = argc > 2 ? std::atoll(argv[2]) : 25;
    const uint32_t w = argc > 3 ? std::atoi(argv[3]) : 18;
    const long long iters_limit = argc > 4 ? std::atoll(argv[4]) : 0;
    const std::string outfile = argc > 5 ? argv[5] : "residue.txt";

    const uint32_t q = n / w, r = n % w;
    const uint32_t Xlen = pad(q + 3);
    const uint32_t P = pad(2 * Xlen + 2);
    uint32_t M = 1, logM = 0;
    while (M < P) { M <<= 1; ++logM; }
    const uint32_t T1len = pad(std::max(q + 1, P - q - 1) + 1);
    std::printf("n=%u c=%lld w=%u  words=%u  FFT M=%u (2^%u complex)\n", n, cc, w, Xlen, M, logM);

    // ---- OpenCL setup
    cl_platform_id plat;
    cl_uint np = 0;
    CHECK(clGetPlatformIDs(1, &plat, &np));
    cl_device_id dev;
    CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr));
    char name[256];
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    std::printf("device: %s\n", name);
    Ctx c{};
    cl_int err;
    c.ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CHECK(err);
    c.q = clCreateCommandQueue(c.ctx, dev, 0, &err);
    CHECK(err);
    std::string exe_dir = argv[0];
    auto pos = exe_dir.find_last_of("\\/");
    exe_dir = pos == std::string::npos ? "" : exe_dir.substr(0, pos + 1);
    std::ifstream kf(exe_dir + "gpuprp.cl");
    if (!kf) { std::fprintf(stderr, "cannot open gpuprp.cl\n"); return 1; }
    std::stringstream ss;
    ss << kf.rdbuf();
    std::string src = ss.str();
    const char* s = src.c_str();
    c.prog = clCreateProgramWithSource(c.ctx, 1, &s, nullptr, &err);
    CHECK(err);
    if (clBuildProgram(c.prog, 1, &dev, "", nullptr, nullptr) != CL_SUCCESS) {
        size_t len;
        clGetProgramBuildInfo(c.prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::string log(len, 0);
        clGetProgramBuildInfo(c.prog, dev, CL_PROGRAM_BUILD_LOG, len, log.data(), nullptr);
        std::fprintf(stderr, "%s\n", log.c_str());
        return 1;
    }
    c.k_load = clCreateKernel(c.prog, "load", &err); CHECK(err);
    c.k_r4 = clCreateKernel(c.prog, "fft_r4", &err); CHECK(err);
    c.k_r2 = clCreateKernel(c.prog, "fft_r2", &err); CHECK(err);
    c.k_sq = clCreateKernel(c.prog, "square", &err); CHECK(err);
    c.k_round = clCreateKernel(c.prog, "round_conv", &err); CHECK(err);
    c.k_cA = clCreateKernel(c.prog, "carryA", &err); CHECK(err);
    c.k_cB = clCreateKernel(c.prog, "carryB", &err); CHECK(err);
    c.k_chk = clCreateKernel(c.prog, "check_zero", &err); CHECK(err);
    c.k_fold = clCreateKernel(c.prog, "fold", &err); CHECK(err);

    // ---- buffers
    std::vector<double> twh(2 * M);
    const double PI = 3.14159265358979323846;
    for (uint32_t m = 0; m < M; ++m) {
        twh[2 * m] = std::cos(2 * PI * m / M);
        twh[2 * m + 1] = -std::sin(2 * PI * m / M);
    }
    auto mk = [&](size_t bytes) {
        cl_mem b = clCreateBuffer(c.ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
        CHECK(err);
        return b;
    };
    cl_mem tw = mk(16ull * M), buf = mk(16ull * M), buf2 = mk(16ull * M);
    cl_mem X = mk(8ull * Xlen), conv = mk(8ull * P), T1 = mk(8ull * T1len);
    const uint32_t maxblk = std::max(P, std::max(T1len, Xlen)) / BLK;
    cl_mem c1 = mk(8ull * maxblk), c2 = mk(8ull * maxblk), c3 = mk(8ull * maxblk);
    cl_mem errb = mk(4), flag = mk(4);
    CHECK(clEnqueueWriteBuffer(c.q, tw, CL_TRUE, 0, 16ull * M, twh.data(), 0, nullptr, nullptr));
    std::vector<long long> xh(Xlen, 0);
    xh[0] = 3;
    CHECK(clEnqueueWriteBuffer(c.q, X, CL_TRUE, 0, 8ull * Xlen, xh.data(), 0, nullptr, nullptr));
    uint32_t zero = 0;
    CHECK(clEnqueueWriteBuffer(c.q, errb, CL_TRUE, 0, 4, &zero, 0, nullptr, nullptr));
    CHECK(clEnqueueWriteBuffer(c.q, flag, CL_TRUE, 0, 4, &zero, 0, nullptr, nullptr));

    auto carry = [&](cl_mem a, uint32_t len) {
        uint32_t nb = len / BLK;
        clSetKernelArg(c.k_cA, 0, sizeof(cl_mem), &a);
        clSetKernelArg(c.k_cA, 1, sizeof(cl_mem), &c1);
        clSetKernelArg(c.k_cA, 2, 4, &nb);
        clSetKernelArg(c.k_cA, 3, 4, &w);
        run1(c, c.k_cA, nb);
        clSetKernelArg(c.k_cB, 0, sizeof(cl_mem), &a);
        clSetKernelArg(c.k_cB, 1, sizeof(cl_mem), &c1);
        clSetKernelArg(c.k_cB, 2, sizeof(cl_mem), &c2);
        clSetKernelArg(c.k_cB, 3, 4, &nb);
        clSetKernelArg(c.k_cB, 4, 4, &w);
        run1(c, c.k_cB, nb);
        clSetKernelArg(c.k_cB, 1, sizeof(cl_mem), &c2);
        clSetKernelArg(c.k_cB, 2, sizeof(cl_mem), &c3);
        run1(c, c.k_cB, nb);
        clSetKernelArg(c.k_chk, 0, sizeof(cl_mem), &c3);
        clSetKernelArg(c.k_chk, 1, 4, &nb);
        clSetKernelArg(c.k_chk, 2, sizeof(cl_mem), &flag);
        run1(c, c.k_chk, nb);
    };
    auto fold = [&](cl_mem D, uint32_t Dlen, cl_mem T, uint32_t Tlen) {
        cl_long cl_cc = cc;
        clSetKernelArg(c.k_fold, 0, sizeof(cl_mem), &D);
        clSetKernelArg(c.k_fold, 1, sizeof(cl_mem), &T);
        clSetKernelArg(c.k_fold, 2, 4, &Dlen);
        clSetKernelArg(c.k_fold, 3, 4, &Tlen);
        clSetKernelArg(c.k_fold, 4, 4, &q);
        clSetKernelArg(c.k_fold, 5, 4, &r);
        clSetKernelArg(c.k_fold, 6, 4, &w);
        clSetKernelArg(c.k_fold, 7, 8, &cl_cc);
        run1(c, c.k_fold, Tlen);
    };
    auto fft = [&](int dir) {  // in: buf, out: buf (ping-pong through buf2)
        cl_mem in = buf, out = buf2;
        uint32_t p = 1;
        int d = dir;
        if (logM & 1) {
            clSetKernelArg(c.k_r2, 0, sizeof(cl_mem), &in);
            clSetKernelArg(c.k_r2, 1, sizeof(cl_mem), &out);
            clSetKernelArg(c.k_r2, 2, sizeof(cl_mem), &tw);
            clSetKernelArg(c.k_r2, 3, 4, &p);
            clSetKernelArg(c.k_r2, 4, 4, &M);
            clSetKernelArg(c.k_r2, 5, 4, &d);
            run1(c, c.k_r2, M / 2);
            std::swap(in, out);
            p *= 2;
        }
        while (p < M) {
            clSetKernelArg(c.k_r4, 0, sizeof(cl_mem), &in);
            clSetKernelArg(c.k_r4, 1, sizeof(cl_mem), &out);
            clSetKernelArg(c.k_r4, 2, sizeof(cl_mem), &tw);
            clSetKernelArg(c.k_r4, 3, 4, &p);
            clSetKernelArg(c.k_r4, 4, 4, &M);
            clSetKernelArg(c.k_r4, 5, 4, &d);
            run1(c, c.k_r4, M / 4);
            std::swap(in, out);
            p *= 4;
        }
        if (in != buf) {
            CHECK(clEnqueueCopyBuffer(c.q, in, buf, 0, 0, 16ull * M, 0, nullptr, nullptr));
        }
    };

    // exponent E = 2^n - c - 1, processed MSB first; we start from x = 3 (top bit)
    auto ebit = [&](uint32_t i) -> int {  // bit i of 2^n - c - 1 (n > 40)
        if (i >= 40) return 1;
        unsigned long long low = ((1ULL << 40) - (unsigned long long)cc - 1ULL);
        return (int)((low >> i) & 1ULL);
    };
    long long total = (long long)n - 1;  // bits n-2 .. 0
    long long todo = iters_limit > 0 ? std::min<long long>(iters_limit, total) : total;
    const double invM = 1.0 / M;

    auto t0 = std::chrono::steady_clock::now();
    for (long long it = 0; it < todo; ++it) {
        const uint32_t bit = (uint32_t)(n - 2 - it);
        cl_long mulc = ebit(bit) ? 3 : 1;
        clSetKernelArg(c.k_load, 0, sizeof(cl_mem), &X);
        clSetKernelArg(c.k_load, 1, sizeof(cl_mem), &buf);
        clSetKernelArg(c.k_load, 2, 4, &Xlen);
        clSetKernelArg(c.k_load, 3, 4, &M);
        run1(c, c.k_load, M);
        fft(+1);
        clSetKernelArg(c.k_sq, 0, sizeof(cl_mem), &buf);
        clSetKernelArg(c.k_sq, 1, 4, &M);
        run1(c, c.k_sq, M);
        fft(-1);
        clSetKernelArg(c.k_round, 0, sizeof(cl_mem), &buf);
        clSetKernelArg(c.k_round, 1, sizeof(cl_mem), &conv);
        clSetKernelArg(c.k_round, 2, 4, &M);
        clSetKernelArg(c.k_round, 3, 4, &P);
        clSetKernelArg(c.k_round, 4, 8, &invM);
        clSetKernelArg(c.k_round, 5, 8, &mulc);
        clSetKernelArg(c.k_round, 6, sizeof(cl_mem), &errb);
        run1(c, c.k_round, P);
        carry(conv, P);
        fold(conv, P, T1, T1len);
        carry(T1, T1len);
        fold(T1, T1len, X, Xlen);
        carry(X, Xlen);
        if ((it + 1) % 20000 == 0 || it + 1 == todo) {
            uint32_t e = 0, f = 0;
            CHECK(clEnqueueReadBuffer(c.q, errb, CL_TRUE, 0, 4, &e, 0, nullptr, nullptr));
            CHECK(clEnqueueReadBuffer(c.q, flag, CL_TRUE, 0, 4, &f, 0, nullptr, nullptr));
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("iter %lld/%lld  %.4f ms/iter  maxerr %.4f  carryflag %u\n", it + 1, todo,
                        1000.0 * el / (it + 1), e / 4294967295.0, f);
            std::fflush(stdout);
            if (f) { std::fprintf(stderr, "carry overflow\n"); return 2; }
            if (e / 4294967295.0 > 0.35) { std::fprintf(stderr, "round-off too large\n"); return 3; }
        }
    }
    clFinish(c.q);
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("done %lld iterations in %.2f s  (%.4f ms/iter)\n", todo, el, 1000.0 * el / todo);
    CHECK(clEnqueueReadBuffer(c.q, X, CL_TRUE, 0, 8ull * Xlen, xh.data(), 0, nullptr, nullptr));
    std::ofstream out(outfile);
    out << n << " " << cc << " " << w << " " << todo << "\n";
    for (uint32_t i = 0; i < Xlen; ++i) out << xh[i] << "\n";
    std::printf("residue digits written to %s\n", outfile.c_str());
    return 0;
}
