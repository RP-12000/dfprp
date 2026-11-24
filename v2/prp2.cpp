// prp2: GPU Fermat PRP (base 3) for N = 2^n - c using a weighted real FFT (no zero padding).
// usage: prp2 n [c=25] [-L log2L] [-iters k] [-prof] [-maxbits b] [-B gec_block] [-nogec] [-fresh] [-report k] [-dev i]
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
        std::printf("usage: prp2 n [c=25] [-L log2L] [-iters k] [-prof] [-maxbits b] [-B gec_block] [-nogec] [-fresh] [-report k] [-dev i]\n");
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
    int GB = 1000;  // Gerbicz block size (check every GB^2 iterations)
    bool gec = true, resume = true;
    long long inject = -1;  // test: corrupt one digit at this iteration
    for (; ai < argc; ++ai) {
        std::string a = argv[ai];
        if (a == "-L" && ai + 1 < argc) logL = std::atoi(argv[++ai]);
        else if (a == "-iters" && ai + 1 < argc) iters_limit = std::atoll(argv[++ai]);
        else if (a == "-prof") prof = true;
        else if (a == "-maxbits" && ai + 1 < argc) maxbits = std::atof(argv[++ai]);
        else if (a == "-dev" && ai + 1 < argc) devIdx = std::atoi(argv[++ai]);
        else if (a == "-report" && ai + 1 < argc) report = std::atoll(argv[++ai]);
        else if (a == "-B" && ai + 1 < argc) GB = std::atoi(argv[++ai]);
        else if (a == "-nogec") gec = false;
        else if (a == "-inject" && ai + 1 < argc) inject = std::atoll(argv[++ai]);
        else if (a == "-fresh") resume = false;
        else { std::fprintf(stderr, "bad arg %s\n", a.c_str()); return 1; }
    }
    if (!logL) { logL = 4; while ((double)n / (double)(1ull << logL) > maxbits) ++logL; }
    const uint64_t L = 1ull << logL, NC = L / 2;
    uint32_t W = 512;
    while (W > 2 && NC / W < 2) W >>= 1;
    const uint32_t H = (uint32_t)(NC / W);
    const uint32_t logW = ilog2(W), logH = ilog2(H);
    uint32_t TGc = std::getenv("PRP2_TG") ? (uint32_t)std::atoi(std::getenv("PRP2_TG")) : 2;
    while (TGc > 1 && (2 * TGc * H > 1024 || 2 * TGc > W / 2)) TGc >>= 1;
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
    char opts[1024];
    const char* extra = std::getenv("PRP2_OPTS");
    std::snprintf(opts, sizeof(opts),
                  "-D W=%u -D H=%u -D LOGW=%u -D LOGH=%u -D LOGL=%d -D NEXP=%uu -D CC=%lldL -D WSPLIT=%u -D TG=%u %s",
                  W, H, logW, logH, logL, n, cc, (unsigned)((logL + 1) / 2), TGc, extra ? extra : "");
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
    cl_kernel kTF = clCreateKernel(prog, "tailFwd", &err); CHECK(err);
    cl_kernel kTM = clCreateKernel(prog, "tailMul", &err); CHECK(err);
    cl_kernel kI = clCreateKernel(prog, "invW", &err); CHECK(err);

    // ---- tables
    std::vector<cplx> twW(W), twH(H), twN1(W), twN2(H);
    for (uint32_t i = 0; i < W; ++i) { twW[i] = root(i, W); twN1[i] = root(i, NC); }
    for (uint32_t i = 0; i < H; ++i) { twH[i] = root(i, H); twN2[i] = root((uint64_t)W * i, NC); }
    const uint32_t WSPLIT = (logL + 1) / 2;
    const size_t TS = (size_t)1 << WSPLIT;
    std::vector<double> WTAB(8 * TS, 1.0);
    {
        const double log2c = std::log2((double)cc);
        for (size_t i = 0; i < TS; ++i) {
            double a = (double)i / (double)L;                    // 2^(r_lo/L)
            double b = (double)(i << WSPLIT) / (double)L;        // 2^(r_hi 2^S / L)
            if (i < TS) { WTAB[i] = std::exp2(a); WTAB[4 * TS + i] = std::exp2(-a); }
            if ((i << WSPLIT) < L) { WTAB[TS + i] = std::exp2(b); WTAB[5 * TS + i] = std::exp2(-b); }
            WTAB[2 * TS + i] = std::exp2(a * log2c);
            WTAB[6 * TS + i] = std::exp2(-a * log2c);
            if ((i << WSPLIT) < L) {
                WTAB[3 * TS + i] = std::exp2(b * log2c);
                WTAB[7 * TS + i] = std::exp2(-b * log2c) / (double)NC;
            }
        }
    }

    auto mk = [&](size_t bytes, const void* init) {
        cl_mem b = clCreateBuffer(ctx, CL_MEM_READ_WRITE | (init ? CL_MEM_COPY_HOST_PTR : 0), bytes,
                                  (void*)init, &err);
        CHECK(err);
        return b;
    };
    cl_mem bTwW = mk(16 * W, twW.data()), bTwH = mk(16 * H, twH.data());
    cl_mem bTwN1 = mk(16 * W, twN1.data()), bTwN2 = mk(16 * H, twN2.data());
    cl_mem bWT = mk(8 * WTAB.size(), WTAB.data()), bIWT = bWT;
    cl_mem bA = mk(16 * NC, nullptr), bA2 = mk(16 * NC, nullptr);
    uint32_t zero = 0;
    cl_mem bErr = mk(4, &zero);

    struct Num { cl_mem D, CY; };
    std::vector<int32_t> Dh(2 * NC, 0);
    std::vector<int64_t> CYh(NC, 0);
    auto mkNum = [&]() { return Num{mk(8 * NC, Dh.data()), mk(8 * NC, CYh.data())}; };
    auto setSmall = [&](Num a, int v) {  // a = v (small)
        std::vector<int32_t> d(2 * NC, 0);
        std::vector<int64_t> cy(NC, 0);
        d[0] = v;
        CHECK(clEnqueueWriteBuffer(q, a.D, CL_TRUE, 0, 8 * NC, d.data(), 0, nullptr, nullptr));
        CHECK(clEnqueueWriteBuffer(q, a.CY, CL_TRUE, 0, 8 * NC, cy.data(), 0, nullptr, nullptr));
    };
    auto copyNum = [&](Num dst, Num src) {
        CHECK(clEnqueueCopyBuffer(q, src.D, dst.D, 0, 0, 8 * NC, 0, nullptr, nullptr));
        CHECK(clEnqueueCopyBuffer(q, src.CY, dst.CY, 0, 0, 8 * NC, 0, nullptr, nullptr));
    };
    Num X = mkNum(), Dg = mkNum(), Xs = mkNum(), Tg = mkNum();

    for (cl_kernel k : {kT, kTF, kTM}) {
        clSetKernelArg(k, 1, sizeof(cl_mem), &bTwH);
        clSetKernelArg(k, 2, sizeof(cl_mem), &bTwN1);
        clSetKernelArg(k, 3, sizeof(cl_mem), &bTwN2);
    }
    clSetKernelArg(kT, 0, sizeof(cl_mem), &bA);
    clSetKernelArg(kTF, 0, sizeof(cl_mem), &bA2);
    clSetKernelArg(kTM, 0, sizeof(cl_mem), &bA);
    clSetKernelArg(kTM, 4, sizeof(cl_mem), &bA2);
    clSetKernelArg(kF, 2, sizeof(cl_mem), &bWT);
    clSetKernelArg(kF, 4, sizeof(cl_mem), &bTwW);
    clSetKernelArg(kF, 5, sizeof(cl_mem), &bTwN1);
    clSetKernelArg(kF, 6, sizeof(cl_mem), &bTwN2);
    clSetKernelArg(kI, 0, sizeof(cl_mem), &bA);
    clSetKernelArg(kI, 3, sizeof(cl_mem), &bIWT);
    clSetKernelArg(kI, 4, sizeof(cl_mem), &bTwW);
    clSetKernelArg(kI, 5, sizeof(cl_mem), &bTwN1);
    clSetKernelArg(kI, 6, sizeof(cl_mem), &bTwN2);
    clSetKernelArg(kI, 8, sizeof(cl_mem), &bErr);

    const size_t WGS = 64;
    const size_t gF = (size_t)H * WGS, gT = (size_t)(W / (2 * TGc)) * WGS, gI = (size_t)H * WGS;
    cl_mem curF = nullptr, curFout = nullptr, curI = nullptr;
    int curMul = -1;
    bool profNow = false;
    double tF = 0, tT = 0, tI = 0;
    long long nprof = 0;
    std::vector<std::pair<cl_event, double*>> pend;
    cl_ulong spanA = 0, spanB = 0;
    auto flushProf = [&]() {
        if (pend.empty()) return;
        clWaitForEvents(1, &pend.back().first);
        for (auto& pe : pend) {
            cl_ulong a, b;
            clGetEventProfilingInfo(pe.first, CL_PROFILING_COMMAND_START, 8, &a, nullptr);
            clGetEventProfilingInfo(pe.first, CL_PROFILING_COMMAND_END, 8, &b, nullptr);
            *pe.second += (b - a) * 1e-6;
            if (!spanA || a < spanA) spanA = a;
            if (b > spanB) spanB = b;
            clReleaseEvent(pe.first);
        }
        pend.clear();
    };
    auto enq = [&](cl_kernel k, size_t g, double* acc) {
        cl_event e;
        CHECK(clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, &WGS, 0, nullptr, profNow ? &e : nullptr));
        if (profNow) {
            pend.push_back({e, acc});
            if (pend.size() >= 600) flushProf();
        }
    };
    auto fwd = [&](Num a, cl_mem out) {
        if (curF != a.D) {
            clSetKernelArg(kF, 0, sizeof(cl_mem), &a.D);
            clSetKernelArg(kF, 1, sizeof(cl_mem), &a.CY);
            curF = a.D;
        }
        if (curFout != out) { clSetKernelArg(kF, 3, sizeof(cl_mem), &out); curFout = out; }
        enq(kF, gF, &tF);
    };
    auto inv = [&](Num a, int mulc) {
        if (curI != a.D) {
            clSetKernelArg(kI, 1, sizeof(cl_mem), &a.D);
            clSetKernelArg(kI, 2, sizeof(cl_mem), &a.CY);
            curI = a.D;
        }
        if (mulc != curMul) { clSetKernelArg(kI, 7, sizeof(int), &mulc); curMul = mulc; }
        enq(kI, gI, &tI);
    };
    auto sqr = [&](Num a, int mulc) {  // a = a^2 * mulc
        fwd(a, bA);
        enq(kT, gT, &tT);
        inv(a, mulc);
    };
    auto mul = [&](Num a, Num b, int mulc) {  // a = a * b * mulc
        fwd(b, bA2);
        enq(kTF, gT, &tT);
        fwd(a, bA);
        enq(kTM, gT, &tT);
        inv(a, mulc);
    };
    cl_event evPrev = nullptr;
    long long enqCount = 0;
    auto throttle = [&]() {
        if ((++enqCount & 255) == 0) {
            cl_event ev;
            CHECK(clEnqueueMarker(q, &ev));
            clFlush(q);
            if (evPrev) { clWaitForEvents(1, &evPrev); clReleaseEvent(evPrev); }
            evPrev = ev;
        }
    };

    // ---- canonical residue (host): X = sum d_j 2^B_j + sum CY_m 2^B_(2m+2), reduced into [0, N)
    const size_t K = (n + 31) / 32 + 1;
    auto Bj = [&](uint64_t j) { return ((uint64_t)n * j + L - 1) >> logL; };
    auto canon = [&](Num a) {
        CHECK(clEnqueueReadBuffer(q, a.D, CL_TRUE, 0, 8 * NC, Dh.data(), 0, nullptr, nullptr));
        CHECK(clEnqueueReadBuffer(q, a.CY, CL_TRUE, 0, 8 * NC, CYh.data(), 0, nullptr, nullptr));
        std::vector<int64_t> acc(K + 3, 0);
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
                addAt(Bj(2 * m), Dh[2 * p]);
                addAt(Bj(2 * m + 1), Dh[2 * p + 1]);
                addAt(Bj(2 * m + 2), CYh[p]);
            }
        std::vector<uint32_t> limb(K, 0);
        int64_t top = 0;
        auto normalize = [&]() {
            int64_t carry = 0;
            for (size_t i = 0; i < K; ++i) {
                int64_t t = acc[i] + carry;
                limb[i] = (uint32_t)(t & 0xFFFFFFFFll);
                carry = t >> 32;
            }
            for (size_t i = K; i < acc.size(); ++i) carry += acc[i] * (1ll << (32 * (i - K)));
            top = carry;
        };
        normalize();
        const uint32_t nl = n / 32, nb = n % 32;
        for (int round = 0; round < 8; ++round) {
            int64_t hi = (int64_t)(limb[nl] >> nb);
            int sh = 32 - nb;
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
            normalize();
        }
        std::vector<uint32_t> t(limb);
        uint64_t carry = (uint64_t)cc;
        for (size_t i = 0; i < K && carry; ++i) {
            uint64_t v = (uint64_t)t[i] + carry;
            t[i] = (uint32_t)v;
            carry = v >> 32;
        }
        if ((t[nl] >> nb) & 1) { t[nl] &= (nb ? ((1u << nb) - 1) : 0u); limb = t; }
        return limb;
    };

    // exponent E = 2^n - c - 1, MSB first, start from x = 3 (top bit)
    auto ebit = [&](uint32_t i) -> int {
        if (i >= 40) return 1;
        unsigned long long low = ((1ULL << 40) - (unsigned long long)cc - 1ULL);
        return (int)((low >> i) & 1ULL);
    };
    const long long total = (long long)n - 1;
    auto mulcOf = [&](long long it) { return ebit((uint32_t)(n - 2 - it)) ? 3 : 1; };

    auto t0 = std::chrono::steady_clock::now();
    auto tl = t0;
    long long itl = 0, itStart = 0;
    long long todo = total;
    auto progress = [&](long long done, const char* tag) {
        uint32_t e2 = 0;
        CHECK(clEnqueueReadBuffer(q, bErr, CL_TRUE, 0, 4, &e2, 0, nullptr, nullptr));
        CHECK(clEnqueueWriteBuffer(q, bErr, CL_TRUE, 0, 4, &zero, 0, nullptr, nullptr));
        auto now = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(now - t0).count();
        double seg = std::chrono::duration<double>(now - tl).count();
        double msit = done > itl ? 1000.0 * seg / (done - itl) : 0;
        std::printf("%s iter %lld/%lld  %.4f ms/iter  maxerr %.4f  ETA %.1f min  (elapsed %.1f s)\n", tag,
                    done, todo, msit, e2 / 4294967295.0, msit * (todo - done) / 60000.0, el);
        std::fflush(stdout);
        tl = now; itl = done;
        if (e2 / 4294967295.0 > 0.42) { std::fprintf(stderr, "round-off too large\n"); std::exit(3); }
    };
    auto printRes = [&](const std::vector<uint32_t>& limb, bool full) {
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
        if (full) std::printf("2^%u-%lld is %s\n", n, cc, isone ? "a probable prime!" : "composite");
    };

    setSmall(X, 3);
    if (!gec || iters_limit > 0) {
        // plain run (benchmark / partial residue check)
        todo = iters_limit > 0 ? std::min<long long>(iters_limit, total) : total;
        for (long long it = 0; it < todo; ++it) {
            profNow = prof && it >= 200 && nprof < 3000;
            sqr(X, mulcOf(it));
            if (profNow) ++nprof;
            throttle();
            if ((it + 1) % report == 0 || it + 1 == todo) progress(it + 1, "plain");
        }
        clFinish(q);
        flushProf();
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("done %lld iterations in %.2f s  (%.4f ms/iter)\n", todo, el, 1000.0 * el / todo);
        if (nprof)
            std::printf("kernel ms (avg of %lld): fwdW %.4f  tail %.4f  invW %.4f  sum %.4f\n", nprof,
                        tF / nprof, tT / nprof, tI / nprof, (tF + tT + tI) / nprof);
        printRes(canon(X), todo == total);
        return 0;
    }

    // ---- full test with Gerbicz check.
    // Iterations 0 .. G-1 all have mulc = 3 (exponent bits >= 40). y = 3x then satisfies y -> y^2,
    // so with d_s = prod_{u<=s} y_u (one y every B iterations): d_s == y_0 * d_(s-1)^(2^B).
    const long long G = (long long)n - 41;
    char ckname[128];
    std::snprintf(ckname, sizeof(ckname), "prp2_%u_%lld.ckpt", n, cc);
    long long it = 0;
    auto saveCk = [&](long long iter) {
        CHECK(clEnqueueReadBuffer(q, X.D, CL_TRUE, 0, 8 * NC, Dh.data(), 0, nullptr, nullptr));
        CHECK(clEnqueueReadBuffer(q, X.CY, CL_TRUE, 0, 8 * NC, CYh.data(), 0, nullptr, nullptr));
        std::string tmp = std::string(ckname) + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) return;
        uint64_t hdr[4] = {n, (uint64_t)cc, (uint64_t)logL, (uint64_t)iter};
        std::fwrite(hdr, 8, 4, f);
        std::fwrite(Dh.data(), 4, Dh.size(), f);
        std::fwrite(CYh.data(), 8, CYh.size(), f);
        std::fclose(f);
        std::remove(ckname);
        std::rename(tmp.c_str(), ckname);
    };
    if (resume) {
        FILE* f = std::fopen(ckname, "rb");
        if (f) {
            uint64_t hdr[4];
            if (std::fread(hdr, 8, 4, f) == 4 && hdr[0] == n && hdr[1] == (uint64_t)cc &&
                hdr[2] == (uint64_t)logL && std::fread(Dh.data(), 4, Dh.size(), f) == Dh.size() &&
                std::fread(CYh.data(), 8, CYh.size(), f) == CYh.size()) {
                CHECK(clEnqueueWriteBuffer(q, X.D, CL_TRUE, 0, 8 * NC, Dh.data(), 0, nullptr, nullptr));
                CHECK(clEnqueueWriteBuffer(q, X.CY, CL_TRUE, 0, 8 * NC, CYh.data(), 0, nullptr, nullptr));
                it = (long long)hdr[3];
                std::printf("resumed from %s at iteration %lld\n", ckname, it);
            }
            std::fclose(f);
        }
    }
    itl = itStart = it;
    int fails = 0;
    const long long S = (long long)GB * GB;
    while (G - it >= GB) {
        const long long blocks = std::min<long long>(S, G - it) / GB;
        copyNum(Xs, X);
        setSmall(Dg, 1);
        mul(Dg, X, 3);  // d_0 = y_0 = 3 x_0
        long long j = it;
        for (long long b = 0; b < blocks; ++b) {
            for (int k = 0; k < GB; ++k, ++j) {
                sqr(X, 3);
                throttle();
                if (j == inject) {
                    int32_t bad = 12345;
                    CHECK(clEnqueueWriteBuffer(q, X.D, CL_TRUE, 40, 4, &bad, 0, nullptr, nullptr));
                    std::printf("injected error at iteration %lld\n", j);
                    inject = -1;
                }
                if ((j + 1) % report == 0) progress(j + 1, "gec");
            }
            if (b == blocks - 1) copyNum(Tg, Dg);  // d_(s-1)
            mul(Dg, X, 3);                          // d_s = d_(s-1) * y_s
            throttle();
        }
        for (int k = 0; k < GB; ++k) { sqr(Tg, 1); throttle(); }
        mul(Tg, Xs, 3);
        clFinish(q);
        bool ok = canon(Tg) == canon(Dg);
        if (ok) {
            it = j;
            saveCk(it);
            std::printf("Gerbicz check OK at iteration %lld\n", it);
        } else {
            ++fails;
            std::printf("Gerbicz check FAILED at iteration %lld (fail #%d), rolling back to %lld\n", j,
                        fails, it);
            if (fails >= 5) { std::fprintf(stderr, "too many errors\n"); return 4; }
            copyNum(X, Xs);
        }
        std::fflush(stdout);
    }
    // remaining iterations (< B not covered by GEC, plus the 40 low bits): run twice and compare
    std::vector<uint32_t> r1, r2;
    copyNum(Xs, X);
    bool agreed = false;
    for (int attempt = 0; attempt < 6 && !agreed; ++attempt) {
        if (attempt) copyNum(X, Xs);
        for (long long j = it; j < total; ++j) { sqr(X, mulcOf(j)); throttle(); }
        clFinish(q);
        auto r = canon(X);
        if ((!r1.empty() && r == r1) || (!r2.empty() && r == r2)) { r1 = r; agreed = true; }
        else if (r1.empty()) r1 = r;
        else r2 = r;
    }
    if (!agreed) { std::fprintf(stderr, "final iterations not reproducible\n"); return 5; }
    progress(total, "final");
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("done %lld iterations in %.2f s (%.4f ms/iter incl. checks)\n", total - itStart, el,
                1000.0 * el / std::max<long long>(1, total - itStart));
    printRes(r1, true);
    std::remove(ckname);
    return 0;
}
