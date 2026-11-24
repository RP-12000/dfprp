// prp4: GPU Fermat PRP (base 3) for N = 2^n - c using an exact integer NTT (three 31-bit primes + CRT).
// usage: prp4 n [c=25] [-L log2L] [-iters k (plain)] [-prof] [-B gec_block] [-nogec] [-fresh] [-report k]
//        [-inject it] [-dev i]
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(e)                                                                \
    do {                                                                        \
        cl_int _e = (e);                                                        \
        if (_e != CL_SUCCESS) {                                                 \
            std::fprintf(stderr, "OpenCL error %d at line %d\n", _e, __LINE__); \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

typedef uint64_t u64;
typedef uint32_t u32;
static u64 mulmod(u64 a, u64 b, u64 p) { return (a * b) % p; }  // p < 2^31 so a*b < 2^62
static u64 powmod(u64 a, u64 e, u64 p) {
    u64 r = 1;
    a %= p;
    while (e) { if (e & 1) r = mulmod(r, a, p); a = mulmod(a, a, p); e >>= 1; }
    return r;
}
static u64 invmod(u64 a, u64 p) { return powmod(a, p - 2, p); }

static const u32 PRIMES[3] = {2013265921u, 2113929217u, 2130706433u};  // 15*2^27+1, 63*2^25+1, 127*2^24+1
static u32 genOf(u32 p) {
    u32 f[8], nf = 0;
    u32 m = p - 1;
    for (u32 d = 2; d * d <= m; ++d)
        if (m % d == 0) { f[nf++] = d; while (m % d == 0) m /= d; }
    if (m > 1) f[nf++] = m;
    for (u32 g = 2;; ++g) {
        bool ok = true;
        for (u32 i = 0; i < nf && ok; ++i) ok = powmod(g, (p - 1) / f[i], p) != 1;
        if (ok) return g;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: prp4 n [c=25] [-L log2L] [-iters k] [-prof] [-B b] [-nogec] [-fresh] [-report k]\n");
        return 1;
    }
    const u32 n = (u32)std::atoi(argv[1]);
    int ai = 2;
    long long cc = 25;
    if (argc > 2 && argv[2][0] != '-') { cc = std::atoll(argv[2]); ai = 3; }
    int logL = 0, devIdx = 0, GB = 1000;
    long long iters_limit = 0, report = 100000, inject = -1;
    bool prof = false, gec = true, resume = true;
    for (; ai < argc; ++ai) {
        std::string a = argv[ai];
        if (a == "-L" && ai + 1 < argc) logL = std::atoi(argv[++ai]);
        else if (a == "-iters" && ai + 1 < argc) iters_limit = std::atoll(argv[++ai]);
        else if (a == "-prof") prof = true;
        else if (a == "-dev" && ai + 1 < argc) devIdx = std::atoi(argv[++ai]);
        else if (a == "-report" && ai + 1 < argc) report = std::atoll(argv[++ai]);
        else if (a == "-B" && ai + 1 < argc) GB = std::atoi(argv[++ai]);
        else if (a == "-nogec") gec = false;
        else if (a == "-inject" && ai + 1 < argc) inject = std::atoll(argv[++ai]);
        else if (a == "-fresh") resume = false;
        else { std::fprintf(stderr, "bad arg %s\n", a.c_str()); return 1; }
    }
    const u32 WN = (n + 31) / 32;  // limbs
    if (!logL) { logL = 6; while ((1u << logL) < 2 * WN) ++logL; }
    const u32 L = 1u << logL;
    if (L < 2 * WN) { std::fprintf(stderr, "L too small\n"); return 1; }
    u32 W = std::getenv("PRP4_W") ? (u32)std::atoi(std::getenv("PRP4_W")) : 512;
    while (W > 4 && L / W < 4) W >>= 1;
    const u32 H = L / W;
    u32 TG = std::getenv("PRP2_TG") ? (u32)std::atoi(std::getenv("PRP2_TG")) : 2;
    while (TG > 1 && (3 * TG * H * 4 > 24576 || W % TG)) TG >>= 1;
    const u32 SEG = std::min<u32>(16, H);
    const u32 NSEG = H / SEG;
    const u32 NCOL = (WN + 5 + H - 1) / H + 1;
    const u64 K = (u64)cc << (32 * (u64)WN - n);
    std::printf("n=%u c=%lld  NTT L=%u (3 primes, 32-bit limbs, %u limbs)  W=%u H=%u TG=%u  K=%llu\n", n, cc, L,
                WN, W, H, TG, (unsigned long long)K);

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
        char name[256];
        clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        std::printf("device: %s\n", name);
    }
    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CHECK(err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, prof ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
    CHECK(err);

    // ---- constants and tables
    const u64 R = (1ull << 32);
    u32 PINV[3], S[3];
    std::vector<u32> twW(3 * 2 * W), twH(3 * 2 * H), twN1(3 * 2 * W), twN2(3 * 2 * H);
    auto mont = [&](u64 x, u32 p) { return (u32)((x % p) * (R % p) % p); };
    for (int i = 0; i < 3; ++i) {
        const u32 p = PRIMES[i];
        u32 inv = 1;  // p^-1 mod 2^32 by Newton
        for (int k = 0; k < 5; ++k) inv *= 2 - p * inv;
        PINV[i] = (u32)(0u - inv);
        const u32 g = genOf(p);
        const u64 wL = powmod(g, (p - 1) / L, p), wLi = invmod(wL, p);
        const u64 wW = powmod(wL, H, p), wWi = invmod(wW, p);
        const u64 wH = powmod(wL, W, p), wHi = invmod(wH, p);
        for (u32 j = 0; j < W; ++j) {
            twW[(i * 2 + 0) * W + j] = mont(powmod(wW, j, p), p);
            twW[(i * 2 + 1) * W + j] = mont(powmod(wWi, j, p), p);
            twN1[(i * 2 + 0) * W + j] = mont(powmod(wL, j, p), p);
            twN1[(i * 2 + 1) * W + j] = mont(powmod(wLi, j, p), p);
        }
        for (u32 j = 0; j < H; ++j) {
            twH[(i * 2 + 0) * H + j] = mont(powmod(wH, j, p), p);
            twH[(i * 2 + 1) * H + j] = mont(powmod(wHi, j, p), p);
            twN2[(i * 2 + 0) * H + j] = mont(powmod(wL, (u64)W * j, p), p);
            twN2[(i * 2 + 1) * H + j] = mont(powmod(wLi, (u64)W * j, p), p);
        }
        const u64 r2 = (R % p) * (R % p) % p;
        S[i] = (u32)(r2 * invmod(L % p, p) % p);
    }
    const u32 p1 = PRIMES[0], p2 = PRIMES[1], p3 = PRIMES[2];
    const u32 C12M = mont(invmod(p1 % p2, p2), p2);
    const u32 C13M = mont(invmod(p1 % p3, p3), p3);
    const u32 C23M = mont(invmod(p2 % p3, p3), p3);
    const u64 P01 = (u64)p1 * p2;

    auto stages = [](u32 P, const char* tag) {
        std::string out;
        u32 ps = 1, rest = P;
        while (rest >= 4) { char b[64]; std::snprintf(b, sizeof(b), "%s(2,%u) ", tag, ps); out += b; ps *= 4; rest /= 4; }
        if (rest == 2) { char b[64]; std::snprintf(b, sizeof(b), "%s(1,%u) ", tag, ps); out += b; }
        return out;
    };
    std::string hdr;
    {
        char b[2048];
        std::snprintf(b, sizeof(b),
                      "#define W %uu\n#define H %uu\n#define WN %uu\n#define KLO %uu\n#define KHI %uu\n"
                      "#define NCOL %uu\n#define SEG %uu\n#define TG %uu\n#define NEXP %uu\n#define CC %lldL\n"
                      "#define P0 %uu\n#define P1 %uu\n#define P2 %uu\n#define PINV0 %uu\n#define PINV1 %uu\n"
                      "#define PINV2 %uu\n#define S0 %uu\n#define S1 %uu\n#define S2 %uu\n#define C12M %uu\n"
                      "#define C13M %uu\n#define C23M %uu\n#define P01LO %uu\n#define P01HI %uu\n",
                      W, H, WN, (u32)K, (u32)(K >> 32), NCOL, SEG, TG, n, cc, p1, p2, p3, PINV[0], PINV[1],
                      PINV[2], S[0], S[1], S[2], C12M, C13M, C23M, (u32)P01, (u32)(P01 >> 32));
        hdr = b;
        hdr += "#define WSTAGES " + stages(W, "STW") + "\n";
        hdr += "#define HSTAGES " + stages(H, "STH") + "\n";
    }
    std::string exe_dir = argv[0];
    auto pos = exe_dir.find_last_of("\\/");
    exe_dir = pos == std::string::npos ? "" : exe_dir.substr(0, pos + 1);
    std::ifstream kf(exe_dir + "prp4.cl");
    if (!kf) { std::fprintf(stderr, "cannot open prp4.cl\n"); return 1; }
    std::stringstream ss;
    ss << kf.rdbuf();
    std::string src = hdr + ss.str();
    const char* s = src.c_str();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &s, nullptr, &err);
    CHECK(err);
    const char* extra = std::getenv("PRP2_OPTS");
    if (clBuildProgram(prog, 1, &dev, extra ? extra : "", nullptr, nullptr) != CL_SUCCESS) {
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
    cl_kernel kC = clCreateKernel(prog, "carry", &err); CHECK(err);

    auto mk = [&](size_t bytes, const void* init) {
        cl_mem b = clCreateBuffer(ctx, CL_MEM_READ_WRITE | (init ? CL_MEM_COPY_HOST_PTR : 0), bytes,
                                  (void*)init, &err);
        CHECK(err);
        return b;
    };
    cl_mem bTwW = mk(4 * twW.size(), twW.data()), bTwH = mk(4 * twH.size(), twH.data());
    cl_mem bTwN1 = mk(4 * twN1.size(), twN1.data()), bTwN2 = mk(4 * twN2.size(), twN2.data());
    cl_mem bA = mk(4ull * 3 * L, nullptr), bA2 = mk(4ull * 3 * L, nullptr), bZ = mk(4ull * 3 * L, nullptr);

    struct Num { cl_mem X, XC; };
    const size_t NXC = (size_t)NSEG * NCOL;
    std::vector<u32> Xh(L, 0), XCh(NXC, 0);
    auto mkNum = [&]() { return Num{mk(4ull * L, Xh.data()), mk(4 * NXC, XCh.data())}; };
    auto setSmall = [&](Num a, u32 v) {
        std::vector<u32> x(L, 0), xc(NXC, 0);
        x[0] = v;
        CHECK(clEnqueueWriteBuffer(q, a.X, CL_TRUE, 0, 4ull * L, x.data(), 0, nullptr, nullptr));
        CHECK(clEnqueueWriteBuffer(q, a.XC, CL_TRUE, 0, 4 * NXC, xc.data(), 0, nullptr, nullptr));
    };
    auto copyNum = [&](Num dst, Num src) {
        CHECK(clEnqueueCopyBuffer(q, src.X, dst.X, 0, 0, 4ull * L, 0, nullptr, nullptr));
        CHECK(clEnqueueCopyBuffer(q, src.XC, dst.XC, 0, 0, 4 * NXC, 0, nullptr, nullptr));
    };
    Num X = mkNum(), Dg = mkNum(), Xs = mkNum(), Tg = mkNum();

    clSetKernelArg(kF, 3, sizeof(cl_mem), &bTwW);
    clSetKernelArg(kF, 4, sizeof(cl_mem), &bTwN1);
    clSetKernelArg(kF, 5, sizeof(cl_mem), &bTwN2);
    clSetKernelArg(kT, 0, sizeof(cl_mem), &bA);
    clSetKernelArg(kT, 1, sizeof(cl_mem), &bTwH);
    clSetKernelArg(kTF, 0, sizeof(cl_mem), &bA2);
    clSetKernelArg(kTF, 1, sizeof(cl_mem), &bTwH);
    clSetKernelArg(kTM, 0, sizeof(cl_mem), &bA);
    clSetKernelArg(kTM, 1, sizeof(cl_mem), &bTwH);
    clSetKernelArg(kTM, 2, sizeof(cl_mem), &bA2);
    clSetKernelArg(kI, 0, sizeof(cl_mem), &bA);
    clSetKernelArg(kI, 1, sizeof(cl_mem), &bZ);
    clSetKernelArg(kI, 2, sizeof(cl_mem), &bTwW);
    clSetKernelArg(kI, 3, sizeof(cl_mem), &bTwN1);
    clSetKernelArg(kI, 4, sizeof(cl_mem), &bTwN2);
    clSetKernelArg(kC, 0, sizeof(cl_mem), &bZ);

    const size_t WGS = 64;
    const size_t gF = (size_t)H * WGS, gT = (size_t)(W / TG) * WGS, gI = (size_t)H * WGS;
    const size_t gC = (NXC + WGS - 1) / WGS * WGS;
    cl_mem curF = nullptr, curFout = nullptr, curC = nullptr;
    int curMul = -1;
    bool profNow = false;
    double tF = 0, tT = 0, tI = 0, tC = 0;
    long long nprof = 0;
    std::vector<std::pair<cl_event, double*>> pend;
    auto flushProf = [&]() {
        if (pend.empty()) return;
        clWaitForEvents(1, &pend.back().first);
        for (auto& pe : pend) {
            cl_ulong a, b;
            clGetEventProfilingInfo(pe.first, CL_PROFILING_COMMAND_START, 8, &a, nullptr);
            clGetEventProfilingInfo(pe.first, CL_PROFILING_COMMAND_END, 8, &b, nullptr);
            *pe.second += (b - a) * 1e-6;
            clReleaseEvent(pe.first);
        }
        pend.clear();
    };
    auto enq = [&](cl_kernel k, size_t g, double* acc) {
        cl_event e;
        CHECK(clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, &WGS, 0, nullptr, profNow ? &e : nullptr));
        if (profNow) {
            pend.push_back({e, acc});
            if (pend.size() >= 800) flushProf();
        }
    };
    auto fwd = [&](Num a, cl_mem out) {
        if (curF != a.X) {
            clSetKernelArg(kF, 0, sizeof(cl_mem), &a.X);
            clSetKernelArg(kF, 1, sizeof(cl_mem), &a.XC);
            curF = a.X;
        }
        if (curFout != out) { clSetKernelArg(kF, 2, sizeof(cl_mem), &out); curFout = out; }
        enq(kF, gF, &tF);
    };
    auto inv = [&](Num a, int mulc) {
        if (mulc != curMul) { u32 m = (u32)mulc; clSetKernelArg(kI, 5, sizeof(u32), &m); curMul = mulc; }
        enq(kI, gI, &tI);
        if (curC != a.X) {
            clSetKernelArg(kC, 1, sizeof(cl_mem), &a.X);
            clSetKernelArg(kC, 2, sizeof(cl_mem), &a.XC);
            curC = a.X;
        }
        enq(kC, gC, &tC);
    };
    auto sqr = [&](Num a, int mulc) { fwd(a, bA); enq(kT, gT, &tT); inv(a, mulc); };
    auto mul = [&](Num a, Num b, int mulc) {
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

    // ---- canonical residue: X = sum_j X[j] 2^(32 j) + lazy carries, reduced into [0, N)
    const size_t KL = WN + 1;
    auto canon = [&](Num a) {
        CHECK(clEnqueueReadBuffer(q, a.X, CL_TRUE, 0, 4ull * L, Xh.data(), 0, nullptr, nullptr));
        CHECK(clEnqueueReadBuffer(q, a.XC, CL_TRUE, 0, 4 * NXC, XCh.data(), 0, nullptr, nullptr));
        const size_t AS = (size_t)WN + 2 * H + 64;
        std::vector<int64_t> acc(AS, 0);
        auto addLimb = [&](u64 j, u64 v) {  // add v (< 2^34) at limb j
            acc[j] += (int64_t)(v & 0xFFFFFFFFull);
            acc[j + 1] += (int64_t)(v >> 32);
        };
        for (u32 j = 0; j < WN + 8 && j < L; ++j) addLimb(j, Xh[(size_t)(j % H) * W + j / H]);
        for (u32 sg = 0; sg < NSEG; ++sg)
            for (u32 col = 0; col < NCOL; ++col) {
                u64 v = XCh[(size_t)sg * NCOL + col];
                if (!v) continue;
                u64 j = sg + 1 < NSEG ? (u64)col * H + (sg + 1) * SEG : (u64)(col + 1) * H;
                addLimb(j, v);
            }
        auto prop = [&]() {
            int64_t carry = 0;
            for (size_t i = 0; i < AS; ++i) {
                int64_t t = acc[i] + carry;
                acc[i] = t & 0xFFFFFFFFll;
                carry = t >> 32;
            }
        };
        prop();
        // fold limbs >= WN: 2^(32 WN) == m = c * 2^(32 WN - n) (mod N), m < 2^37
        const u64 m = (u64)cc << (32 * (u64)WN - n);
        for (int round = 0; round < 16; ++round) {
            bool any = false;
            for (size_t i = WN; i < AS; ++i) if (acc[i]) { any = true; break; }
            if (!any) break;
            std::vector<int64_t> hi(acc.begin() + WN, acc.end());
            std::fill(acc.begin() + WN, acc.end(), 0);
            for (size_t i = 0; i < hi.size(); ++i) {
                if (!hi[i]) continue;
                u64 v = (u64)hi[i];  // < 2^32 after prop
                u64 lo = v * (m & 0xFFFFFFFFull);
                acc[i] += (int64_t)(lo & 0xFFFFFFFFull);
                acc[i + 1] += (int64_t)(lo >> 32);
                u64 x1 = v * (m >> 32);
                acc[i + 1] += (int64_t)(x1 & 0xFFFFFFFFull);
                acc[i + 2] += (int64_t)(x1 >> 32);
            }
            prop();
        }
        // now value < 2^(32 WN); reduce bits >= n
        std::vector<u32> limb(KL, 0);
        for (size_t i = 0; i < WN; ++i) limb[i] = (u32)acc[i];
        int64_t top = 0;
        auto normalize = [&]() {
            int64_t carry = 0;
            for (size_t i = 0; i < KL; ++i) {
                int64_t t = acc[i] + carry;
                limb[i] = (u32)(t & 0xFFFFFFFFll);
                carry = t >> 32;
            }
            top = carry;
        };
        const u32 nl = n / 32, nb = n % 32;
        for (int round = 0; round < 8; ++round) {
            int64_t hi = (int64_t)(limb[nl] >> nb);
            int sh = 32 - nb;
            for (size_t i = nl + 1; i < KL; ++i) { hi += (int64_t)limb[i] << sh; sh += 32; }
            hi += top * (1ll << sh);
            if (hi == 0) break;
            limb[nl] &= (nb ? ((1u << nb) - 1) : 0u);
            for (size_t i = nl + 1; i < KL; ++i) limb[i] = 0;
            std::fill(acc.begin(), acc.end(), 0);
            for (size_t i = 0; i < KL; ++i) acc[i] = limb[i];
            int64_t add = hi * cc;
            acc[0] += add & 0xFFFFFFFFll;
            acc[1] += add >> 32;
            normalize();
        }
        std::vector<u32> t(limb);
        u64 carry = (u64)cc;
        for (size_t i = 0; i < KL && carry; ++i) {
            u64 v = (u64)t[i] + carry;
            t[i] = (u32)v;
            carry = v >> 32;
        }
        if ((t[nl] >> nb) & 1) { t[nl] &= (nb ? ((1u << nb) - 1) : 0u); limb = t; }
        return limb;
    };

    auto ebit = [&](u32 i) -> int {
        if (i >= 40) return 1;
        unsigned long long low = ((1ULL << 40) - (unsigned long long)cc - 1ULL);
        return (int)((low >> i) & 1ULL);
    };
    const long long total = (long long)n - 1;
    auto mulcOf = [&](long long it) { return ebit((u32)(n - 2 - it)) ? 3 : 1; };

    auto t0 = std::chrono::steady_clock::now();
    auto tl = t0;
    long long itl = 0, itStart = 0, todo = total;
    auto progress = [&](long long done, const char* tag) {
        clFinish(q);
        auto now = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(now - t0).count();
        double sg = std::chrono::duration<double>(now - tl).count();
        double msit = done > itl ? 1000.0 * sg / (done - itl) : 0;
        std::printf("%s iter %lld/%lld  %.4f ms/iter  ETA %.1f min  (elapsed %.1f s)\n", tag, done, todo, msit,
                    msit * (todo - done) / 60000.0, el);
        std::fflush(stdout);
        tl = now; itl = done;
    };
    auto printRes = [&](const std::vector<u32>& limb, bool full) {
        bool isone = limb[0] == 1;
        for (size_t i = 1; i < KL && isone; ++i) isone = limb[i] == 0;
        unsigned long long res64 = (unsigned long long)limb[0] | ((unsigned long long)limb[1] << 32);
        std::printf("RES64 %016llX\n", res64);
        std::string r2048;
        for (int i = 63; i >= 0; --i) {
            char b[9];
            std::snprintf(b, sizeof(b), "%08X", i < (int)KL ? limb[i] : 0u);
            r2048 += b;
        }
        std::printf("RES2048 %s\n", r2048.c_str());
        if (full) std::printf("2^%u-%lld is %s\n", n, cc, isone ? "a probable prime!" : "composite");
    };

    setSmall(X, 3);
    if (!gec || iters_limit > 0) {
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
            std::printf("kernel ms (avg of %lld): fwdW %.4f  tail %.4f  invW %.4f  carry %.4f  sum %.4f\n", nprof,
                        tF / nprof, tT / nprof, tI / nprof, tC / nprof, (tF + tT + tI + tC) / nprof);
        printRes(canon(X), todo == total);
        return 0;
    }

    // ---- full test with Gerbicz check (see prp2 for the derivation)
    const long long G = (long long)n - 41;
    char ckname[128];
    std::snprintf(ckname, sizeof(ckname), "prp4_%u_%lld.ckpt", n, cc);
    long long it = 0;
    auto saveCk = [&](long long iter) {
        CHECK(clEnqueueReadBuffer(q, X.X, CL_TRUE, 0, 4ull * L, Xh.data(), 0, nullptr, nullptr));
        CHECK(clEnqueueReadBuffer(q, X.XC, CL_TRUE, 0, 4 * NXC, XCh.data(), 0, nullptr, nullptr));
        std::string tmp = std::string(ckname) + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) return;
        u64 hd[4] = {n, (u64)cc, (u64)L, (u64)iter};
        std::fwrite(hd, 8, 4, f);
        std::fwrite(Xh.data(), 4, Xh.size(), f);
        std::fwrite(XCh.data(), 4, XCh.size(), f);
        std::fclose(f);
        std::remove(ckname);
        std::rename(tmp.c_str(), ckname);
    };
    if (resume) {
        FILE* f = std::fopen(ckname, "rb");
        if (f) {
            u64 hd[4];
            if (std::fread(hd, 8, 4, f) == 4 && hd[0] == n && hd[1] == (u64)cc && hd[2] == L &&
                std::fread(Xh.data(), 4, Xh.size(), f) == Xh.size() &&
                std::fread(XCh.data(), 4, XCh.size(), f) == XCh.size()) {
                CHECK(clEnqueueWriteBuffer(q, X.X, CL_TRUE, 0, 4ull * L, Xh.data(), 0, nullptr, nullptr));
                CHECK(clEnqueueWriteBuffer(q, X.XC, CL_TRUE, 0, 4 * NXC, XCh.data(), 0, nullptr, nullptr));
                it = (long long)hd[3];
                std::printf("resumed from %s at iteration %lld\n", ckname, it);
            }
            std::fclose(f);
        }
    }
    itl = itStart = it;
    int fails = 0;
    const long long Sg = (long long)GB * GB;
    while (G - it >= GB) {
        const long long blocks = std::min<long long>(Sg, G - it) / GB;
        copyNum(Xs, X);
        setSmall(Dg, 1);
        mul(Dg, X, 3);
        long long j = it;
        for (long long b = 0; b < blocks; ++b) {
            for (int k = 0; k < GB; ++k, ++j) {
                sqr(X, 3);
                throttle();
                if (j == inject) {
                    u32 bad = 12345;
                    CHECK(clEnqueueWriteBuffer(q, X.X, CL_TRUE, 4 * 3, 4, &bad, 0, nullptr, nullptr));
                    std::printf("injected error at iteration %lld\n", j);
                    inject = -1;
                }
                if ((j + 1) % report == 0) progress(j + 1, "gec");
            }
            if (b == blocks - 1) copyNum(Tg, Dg);
            mul(Dg, X, 3);
            throttle();
        }
        for (int k = 0; k < GB; ++k) { sqr(Tg, 1); throttle(); }
        mul(Tg, Xs, 3);
        clFinish(q);
        if (canon(Tg) == canon(Dg)) {
            it = j;
            saveCk(it);
            std::printf("Gerbicz check OK at iteration %lld\n", it);
        } else {
            ++fails;
            std::printf("Gerbicz check FAILED at iteration %lld (fail #%d), rolling back to %lld\n", j, fails, it);
            if (fails >= 5) { std::fprintf(stderr, "too many errors\n"); return 4; }
            copyNum(X, Xs);
        }
        std::fflush(stdout);
    }
    std::vector<u32> r1, r2;
    copyNum(Xs, X);
    bool agreed = false;
    for (int attempt = 0; attempt < 6 && !agreed; ++attempt) {
        if (attempt) copyNum(X, Xs);
        for (long long jj = it; jj < total; ++jj) { sqr(X, mulcOf(jj)); throttle(); }
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
