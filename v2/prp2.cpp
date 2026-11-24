// prp2: GPU Fermat PRP (base 3) for N = 2^n - c using a weighted real FFT (no zero padding).
// usage: prp2 n [c=25] [-L log2L] [-iters k] [-prof] [-maxbits b] [-dev i]
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(e)                                                                    \
    do {                                                                            \
        cl_int _e = (e);                                                            \
        if (_e != CL_SUCCESS) {                                                     \
            std::fprintf(stderr, "OpenCL error %d at line %d\n", _e, __LINE__);     \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

struct cplx { double x, y; };
static const double PI = 3.14159265358979323846;

// exp(-2 pi i j / P), using octant symmetry for accuracy
static cplx root(uint64_t j, uint64_t P) {
    j %= P;
    // reduce to angle in [0, 2pi): use symmetry on j/P
    // compute with quarter-turn reduction
    uint64_t q = (j * 4) / P;          // quadrant
    uint64_t r = j * 4 - q * P;        // remainder, angle within quadrant = (pi/2) * r / P
    double c, s;
    if (2 * r <= P) {
        double a = (PI / 2) * (double)r / (double)P;
        c = std::cos(a); s = std::sin(a);
    } else {
        double a = (PI / 2) * (double)(P - r) / (double)P;
        c = std::sin(a); s = std::cos(a);
    }
    // angle = q*pi/2 + theta ; (c, s) = (cos theta, sin theta)
    double C, S;
    switch (q) {
        case 0: C = c; S = s; break;
        case 1: C = -s; S = c; break;
        case 2: C = -c; S = -s; break;
        default: C = s; S = -c; break;
    }
    return {C, -S};
}

static uint32_t ilog2(uint64_t v) { uint32_t l = 0; while ((1ull << l) < v) ++l; return l; }

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: prp2 n [c=25] [-L log2L] [-iters k] [-prof] [-maxbits b] [-dev i]\n");
        return 1;
    }
    const uint32_t n = (uint32_t)std::atoi(argv[1]);
    int ai = 2;
    long long cc = 25;
    if (argc > 2 && argv[2][0] != '-') { cc = std::atoll(argv[2]); ai = 3; }
    int logL = 0;
    long long iters_limit = 0;
    bool prof = false;
    double maxbits = 15.8;
    int devIdx = 0;
    long long report = 100000;
    for (; ai < argc; ++ai) {
        std::string a = argv[ai];
        if (a == "-L" && ai + 1 < argc) logL = std::atoi(argv[++ai]);
        else if (a == "-iters" && ai + 1 < argc) iters_limit = std::atoll(argv[++ai]);
        else if (a == "-prof") prof = true;
        else if (a == "-maxbits" && ai + 1 < argc) maxbits = std::atof(argv[++ai]);
        else if (a == "-dev" && ai + 1 < argc) devIdx = std::atoi(argv[++ai]);
        else if (a == "-report" && ai + 1 < argc) report = std::atoll(argv[++ai]);
        else { std::fprintf(stderr, "bad arg %s\n", a.c_str()); return 1; }
    }
    if (!logL) { logL = 4; while ((double)n / (double)(1ull << logL) > maxbits) ++logL; }
    const uint64_t L = 1ull << logL, NC = L / 2;
    uint32_t W = 512;
    while (W > 2 && NC / W < 2) W >>= 1;
    const uint32_t H = (uint32_t)(NC / W);
    const uint32_t logW = ilog2(W), logH = ilog2(H);
    std::printf("n=%u c=%lld  L=%llu (2^%d real, %.3f bits/word)  W=%u H=%u\n", n, cc,
                (unsigned long long)L, logL, (double)n / L, W, H);

    // ---- OpenCL setup
    cl_uint np = 0;
    CHECK(clGetPlatformIDs(0, nullptr, &np));
    std::vector<cl_platform_id> plats(np);
    CHECK(clGetPlatformIDs(np, plats.data(), nullptr));
    std::vector<cl_device_id> devs;
    for (auto p : plats) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS) continue;
        std::vector<cl_device_id> d(nd);
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, d.data(), nullptr);
        devs.insert(devs.end(), d.begin(), d.end());
    }
    if (devs.empty()) { std::fprintf(stderr, "no GPU\n"); return 1; }
    cl_device_id dev = devs[std::min<size_t>(devIdx, devs.size() - 1)];
    {
        char name[256], ver[256], drv[256];
        cl_uint cus, clk;
        cl_ulong lmem, gmem;
        clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(ver), ver, nullptr);
        clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof(drv), drv, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clk), &clk, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, nullptr);
        std::printf("device: %s | %s | driver %s | %u CU @ %u MHz | local %llu KB | global %llu MB\n",
                    name, ver, drv, cus, clk, (unsigned long long)lmem / 1024,
                    (unsigned long long)gmem >> 20);
    }
    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CHECK(err);
    cl_command_queue q =
        clCreateCommandQueue(ctx, dev, prof ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
    CHECK(err);
    std::string exe_dir = argv[0];
    auto pos = exe_dir.find_last_of("\\/");
    exe_dir = pos == std::string::npos ? "" : exe_dir.substr(0, pos + 1);
    std::ifstream kf(exe_dir + "prp2.cl");
    if (!kf) { std::fprintf(stderr, "cannot open prp2.cl\n"); return 1; }
    std::stringstream ss;
    ss << kf.rdbuf();
    std::string src = ss.str();
    const char* s = src.c_str();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &s, nullptr, &err);
    CHECK(err);
    char opts[512];
    std::snprintf(opts, sizeof(opts),
                  "-D W=%u -D H=%u -D LOGW=%u -D LOGH=%u -D LOGL=%d -D NEXP=%uu -D CC=%lldL",
                  W, H, logW, logH, logL, n, cc);
    if (clBuildProgram(prog, 1, &dev, opts, nullptr, nullptr) != CL_SUCCESS) {
        size_t len;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::string log(len, 0);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, len, log.data(), nullptr);
        std::fprintf(stderr, "%s\n", log.c_str());
        return 1;
    }
    cl_kernel kF = clCreateKernel(prog, "fwdW", &err); CHECK(err);
    cl_kernel kT = clCreateKernel(prog, "tail", &err); CHECK(err);
    cl_kernel kI = clCreateKernel(prog, "invW", &err); CHECK(err);

    // ---- tables
    std::vector<cplx> twW(W), twH(H), twN1(W), twN2(H);
    for (uint32_t i = 0; i < W; ++i) { twW[i] = root(i, W); twN1[i] = root(i, NC); }
    for (uint32_t i = 0; i < H; ++i) { twH[i] = root(i, H); twN2[i] = root((uint64_t)W * i, NC); }
    std::vector<cplx> WT(NC), IWT(NC);
    const double log2c = std::log2((double)cc);
    auto weight = [&](uint64_t j) {
        uint64_t r = ((uint64_t)n * j) % L;  // frac = ((L - r) mod L)/L
        double frac = (double)((L - r) % L) / (double)L;
        return std::exp2(frac + (double)j * log2c / (double)L);
    };
    for (uint32_t h = 0; h < H; ++h)
        for (uint32_t w = 0; w < W; ++w) {
            uint64_t m = h + (uint64_t)H * w, p = (uint64_t)h * W + w;
            double a = weight(2 * m), b = weight(2 * m + 1);
            WT[p] = {a, b};
            IWT[p] = {1.0 / (a * NC), 1.0 / (b * NC)};
        }

    auto mk = [&](size_t bytes, const void* init) {
        cl_mem b = clCreateBuffer(ctx, CL_MEM_READ_WRITE | (init ? CL_MEM_COPY_HOST_PTR : 0), bytes,
                                  (void*)init, &err);
        CHECK(err);
        return b;
    };
    cl_mem bTwW = mk(16 * W, twW.data()), bTwH = mk(16 * H, twH.data());
    cl_mem bTwN1 = mk(16 * W, twN1.data()), bTwN2 = mk(16 * H, twN2.data());
    cl_mem bWT = mk(16 * NC, WT.data()), bIWT = mk(16 * NC, IWT.data());
    std::vector<int32_t> Dh(2 * NC, 0);
    std::vector<int64_t> CYh(NC, 0);
    Dh[0] = 3;
    cl_mem bD = mk(8 * NC, Dh.data()), bCY = mk(8 * NC, CYh.data()), bA = mk(16 * NC, nullptr);
    uint32_t zero = 0;
    cl_mem bErr = mk(4, &zero);

    clSetKernelArg(kF, 0, sizeof(cl_mem), &bD);
    clSetKernelArg(kF, 1, sizeof(cl_mem), &bCY);
    clSetKernelArg(kF, 2, sizeof(cl_mem), &bWT);
    clSetKernelArg(kF, 3, sizeof(cl_mem), &bA);
    clSetKernelArg(kF, 4, sizeof(cl_mem), &bTwW);
    clSetKernelArg(kF, 5, sizeof(cl_mem), &bTwN1);
    clSetKernelArg(kF, 6, sizeof(cl_mem), &bTwN2);
    clSetKernelArg(kT, 0, sizeof(cl_mem), &bA);
    clSetKernelArg(kT, 1, sizeof(cl_mem), &bTwH);
    clSetKernelArg(kT, 2, sizeof(cl_mem), &bTwN1);
    clSetKernelArg(kT, 3, sizeof(cl_mem), &bTwN2);
    clSetKernelArg(kI, 0, sizeof(cl_mem), &bA);
    clSetKernelArg(kI, 1, sizeof(cl_mem), &bD);
    clSetKernelArg(kI, 2, sizeof(cl_mem), &bCY);
    clSetKernelArg(kI, 3, sizeof(cl_mem), &bIWT);
    clSetKernelArg(kI, 4, sizeof(cl_mem), &bTwW);
    clSetKernelArg(kI, 5, sizeof(cl_mem), &bTwN1);
    clSetKernelArg(kI, 6, sizeof(cl_mem), &bTwN2);
    clSetKernelArg(kI, 8, sizeof(cl_mem), &bErr);
    int curMul = -1;

    const size_t WGS = 64;
    const size_t gF = (size_t)H * WGS, gT = (size_t)(W / 2) * WGS, gI = (size_t)H * WGS;

    // exponent E = 2^n - c - 1, MSB first, start from x = 3 (top bit)
    auto ebit = [&](uint32_t i) -> int {
        if (i >= 40) return 1;
        unsigned long long low = ((1ULL << 40) - (unsigned long long)cc - 1ULL);
        return (int)((low >> i) & 1ULL);
    };
    const long long total = (long long)n - 1;
    const long long todo = iters_limit > 0 ? std::min<long long>(iters_limit, total) : total;

    double tF = 0, tT = 0, tI = 0;
    long long nprof = 0;
    cl_event evPrev = nullptr;
    auto t0 = std::chrono::steady_clock::now();
    auto tl = t0;
    long long itl = 0;
    for (long long it = 0; it < todo; ++it) {
        const uint32_t bit = (uint32_t)(n - 2 - it);
        int mulc = ebit(bit) ? 3 : 1;
        if (mulc != curMul) { clSetKernelArg(kI, 7, sizeof(int), &mulc); curMul = mulc; }
        cl_event e[3];
        bool p = prof && it >= 20 && nprof < 2000;
        CHECK(clEnqueueNDRangeKernel(q, kF, 1, nullptr, &gF, &WGS, 0, nullptr, p ? &e[0] : nullptr));
        CHECK(clEnqueueNDRangeKernel(q, kT, 1, nullptr, &gT, &WGS, 0, nullptr, p ? &e[1] : nullptr));
        CHECK(clEnqueueNDRangeKernel(q, kI, 1, nullptr, &gI, &WGS, 0, nullptr, p ? &e[2] : nullptr));
        if (p) {
            clWaitForEvents(3, e);
            double tt[3];
            for (int k = 0; k < 3; ++k) {
                cl_ulong a, b;
                clGetEventProfilingInfo(e[k], CL_PROFILING_COMMAND_START, 8, &a, nullptr);
                clGetEventProfilingInfo(e[k], CL_PROFILING_COMMAND_END, 8, &b, nullptr);
                tt[k] = (b - a) * 1e-6;
                clReleaseEvent(e[k]);
            }
            tF += tt[0]; tT += tt[1]; tI += tt[2]; ++nprof;
        }
        if ((it & 255) == 255) {
            cl_event ev;
            CHECK(clEnqueueMarker(q, &ev));
            clFlush(q);
            if (evPrev) { clWaitForEvents(1, &evPrev); clReleaseEvent(evPrev); }
            evPrev = ev;
        }
        if ((it + 1) % report == 0 || it + 1 == todo) {
            uint32_t e2 = 0;
            CHECK(clEnqueueReadBuffer(q, bErr, CL_TRUE, 0, 4, &e2, 0, nullptr, nullptr));
            CHECK(clEnqueueWriteBuffer(q, bErr, CL_TRUE, 0, 4, &zero, 0, nullptr, nullptr));
            auto now = std::chrono::steady_clock::now();
            double el = std::chrono::duration<double>(now - t0).count();
            double seg = std::chrono::duration<double>(now - tl).count();
            double msit = 1000.0 * seg / (it + 1 - itl);
            std::printf("iter %lld/%lld  %.4f ms/iter  maxerr %.4f  ETA %.1f min  (elapsed %.1f s)\n",
                        it + 1, todo, msit, e2 / 4294967295.0, msit * (todo - it - 1) / 60000.0, el);
            std::fflush(stdout);
            tl = now; itl = it + 1;
            if (e2 / 4294967295.0 > 0.42) { std::fprintf(stderr, "round-off too large\n"); return 3; }
        }
    }
    clFinish(q);
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("done %lld iterations in %.2f s  (%.4f ms/iter)\n", todo, el, 1000.0 * el / todo);
    if (nprof)
        std::printf("kernel ms (avg of %lld): fwdW %.4f  tail %.4f  invW %.4f  sum %.4f\n", nprof,
                    tF / nprof, tT / nprof, tI / nprof, (tF + tT + tI) / nprof);

    // ---- residue: X = sum d_j 2^B_j + sum CY_m 2^B_(2m+2)
    CHECK(clEnqueueReadBuffer(q, bD, CL_TRUE, 0, 8 * NC, Dh.data(), 0, nullptr, nullptr));
    CHECK(clEnqueueReadBuffer(q, bCY, CL_TRUE, 0, 8 * NC, CYh.data(), 0, nullptr, nullptr));
    const size_t K = (n + 31) / 32 + 1;
    std::vector<int64_t> acc(K + 3, 0);
    auto B = [&](uint64_t j) { return ((uint64_t)n * j + L - 1) >> logL; };
    auto addAt = [&](uint64_t off, int64_t v) {
        size_t i = off >> 5;
        int sh = off & 31;
        int64_t lo = v & 0xFFFFFFFFll, hi = v >> 32;
        uint64_t los = (uint64_t)lo << sh;
        acc[i] += (int64_t)(los & 0xFFFFFFFFull);
        acc[i + 1] += (int64_t)(los >> 32);
        int64_t hs = hi * (1ll << sh);
        acc[i + 1] += hs & 0xFFFFFFFFll;
        acc[i + 2] += hs >> 32;
    };
    for (uint32_t h = 0; h < H; ++h)
        for (uint32_t w = 0; w < W; ++w) {
            uint64_t m = h + (uint64_t)H * w, p = (uint64_t)h * W + w;
            addAt(B(2 * m), Dh[2 * p]);
            addAt(B(2 * m + 1), Dh[2 * p + 1]);
            addAt(B(2 * m + 2), CYh[p]);
        }
    std::vector<uint32_t> limb(K, 0);
    // fold anything beyond K limbs into a signed top carry
    auto normalize = [&](int64_t& top) {
        int64_t carry = 0;
        for (size_t i = 0; i < K; ++i) {
            int64_t t = acc[i] + carry;
            limb[i] = (uint32_t)(t & 0xFFFFFFFFll);
            carry = t >> 32;
        }
        for (size_t i = K; i < acc.size(); ++i) carry += acc[i] * (1ll << (32 * (i - K)));
        top = carry;
    };
    int64_t top;
    normalize(top);
    const uint32_t nl = n / 32, nb = n % 32;  // bit n lies in limb nl at position nb
    for (int round = 0; round < 8; ++round) {
        // Hi = floor(X / 2^n) (signed), clear bits >= n
        int64_t hi = 0;
        // bits from n up to 32K
        int sh = 0;
        hi = (int64_t)(limb[nl] >> nb);
        sh = 32 - nb;
        for (size_t i = nl + 1; i < K; ++i) { hi += (int64_t)limb[i] << sh; sh += 32; }
        hi += top * (1ll << sh);
        if (hi == 0) break;
        limb[nl] &= (nb ? ((1u << nb) - 1) : 0u);
        for (size_t i = nl + 1; i < K; ++i) limb[i] = 0;
        std::fill(acc.begin(), acc.end(), 0);
        for (size_t i = 0; i < K; ++i) acc[i] = limb[i];
        int64_t add = hi * cc;
        acc[0] += add & 0xFFFFFFFFll;
        acc[1] += add >> 32;
        normalize(top);
    }
    // now 0 <= X < 2^n ; if X >= N = 2^n - c, X -= N (i.e. X + c - 2^n)
    bool ge = true;
    {
        // X >= 2^n - c  <=>  X + c >= 2^n
        std::vector<uint32_t> t(limb);
        uint64_t carry = (uint64_t)cc;
        for (size_t i = 0; i < K && carry; ++i) {
            uint64_t v = (uint64_t)t[i] + carry;
            t[i] = (uint32_t)v;
            carry = v >> 32;
        }
        ge = (t[nl] >> nb) & 1;
        if (ge) { t[nl] &= (nb ? ((1u << nb) - 1) : 0u); limb = t; }
    }
    bool isone = limb[0] == 1;
    for (size_t i = 1; i < K && isone; ++i) isone = limb[i] == 0;
    unsigned long long res64 = (unsigned long long)limb[0] | ((unsigned long long)limb[1] << 32);
    std::printf("RES64 %016llX\n", res64);
    std::string r2048;
    for (int i = 63; i >= 0; --i) {
        char b[9];
        std::snprintf(b, sizeof(b), "%08X", i < (int)K ? limb[i] : 0u);
        r2048 += b;
    }
    std::printf("RES2048 %s\n", r2048.c_str());
    if (todo == total) std::printf("2^%u-%lld is %s\n", n, cc, isone ? "a probable prime!" : "composite");
    return 0;
}
