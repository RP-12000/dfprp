// hyprp: GPU Fermat PRP (base 3) for N = 2^n - c, with a weighted real FFT (IBDWT) where each transform is done
// twice in one pass: in plain FP32 (approximate) and exactly over GF(p^2), p = 2^31 - 1 (M31). The exact
// convolution value is rebuilt from the M31 residue and the FP32 approximation, so ~16-18 bits per word fit.
// FFT length L = 2^k. Gerbicz error check; checkpoint/resume.
// usage: hyprp n [c=25] [-L log2L] [-iters k] [-report k] [-prof] [-B gec_block] [-nogec] [-fresh] [-dev i]
//        hyprp -bench [-dev i] [-quick] [-out file.json]
// env:   HYPRP_BACKEND=opencl (use OpenCL even on NVIDIA when built with HYPRP_CUDA)
// build: OpenCL only, or add -DHYPRP_CUDA and link -lcuda -lnvrtc to run on NVIDIA GPUs through CUDA
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
#include <utility>
#include <vector>
#include <memory>

#include "gpu_backend.h"

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

// ---- GF(p) and GF(p^2), p = 2^31 - 1
static const uint64_t P31 = 2147483647ull;
static uint64_t mmod(uint64_t a, uint64_t b) { return (a % P31) * (b % P31) % P31; }
static uint64_t mpow(uint64_t a, uint64_t e) {
    uint64_t r = 1;
    a %= P31;
    while (e) { if (e & 1) r = mmod(r, a); a = mmod(a, a); e >>= 1; }
    return r;
}
struct G2 { uint64_t re, im; };
static G2 gmul(G2 a, G2 b) {
    return G2{(mmod(a.re, b.re) + P31 - mmod(a.im, b.im)) % P31, (mmod(a.re, b.im) + mmod(a.im, b.re)) % P31};
}
static G2 gpow(G2 a, uint64_t e) {
    G2 r{1, 0};
    while (e) { if (e & 1) r = gmul(r, a); a = gmul(a, a); e >>= 1; }
    return r;
}
static bool geq(G2 a, G2 b) { return a.re == b.re && a.im == b.im; }
// primitive 2^K-th root of unity g in GF(p^2) with g^(2^(K-3)) = (1 - i)/sqrt2 = (1 - i) 2^15 (so that the
// radix-4/8 butterflies, written for w^(P/4) = -i, match the tables)
static G2 m31root(uint32_t K) {
    const G2 one{1, 0}, t8{32768, P31 - 32768};
    for (uint64_t a = 2;; ++a) {
        G2 y = gpow(G2{a, 1}, 1073741823ull);  // (p^2 - 1) / 2^32
        G2 z = y;
        for (int i = 0; i < 31; ++i) z = gmul(z, z);
        if (geq(z, one)) continue;  // y must have order exactly 2^32
        G2 g = gpow(y, 1ull << (32 - K));
        G2 w8 = gpow(g, 1ull << (K - 3));
        for (uint64_t e = 1; e < 8; e += 2)
            if (geq(gpow(w8, e), t8)) return gpow(g, e);
    }
}
static int64_t invmod(int64_t a, int64_t m) {  // a^-1 mod m, gcd(a, m) = 1
    int64_t g = m, x = 0, x1 = 1, b = a % m;
    if (b < 0) b += m;
    while (b) { int64_t qq = g / b, t = g - qq * b; g = b; b = t; t = x - qq * x1; x = x1; x1 = t; }
    return ((x % m) + m) % m;
}

static uint32_t ilog2(uint64_t v) { uint32_t l = 0; while ((1ull << l) < v) ++l; return l; }

// largest bits/word for FFT length 2^k (see the length choice in main); 0 = length not supported
static double maxBitsFor(uint32_t k) {
    return k <= 17 ? 17.7 : k == 18 ? 17.45 : k == 19 ? 17.2 : k == 20 ? 16.95 : k == 21 ? 16.5 : 0;
}
// n within 32 of a multiple of L/4: more round-off, the next length is used
static bool nearFraction(uint64_t n, uint64_t L) {
    if (L < 4096) return false;
    const uint64_t q = L / 4, r = n % q;
    return r < 32 || q - r < 32;
}

// ---- benchmark: hyprp -bench [-dev i] [-quick] [-out file]
// Measures this GPU for every FFT length 16K..2048K. Within one length the time per iteration depends on how digits
// are stored, which depends on n:
//   mode 0: 16-bit digits   n <= 16 L
//   mode 1: 32-bit digits   16 L < n <= the length's cutoff
// so each (length, mode) is measured at an n inside its range, by running this program with -L k, two rounds, taking
// the faster ms/iter of the last quarter of each run (steady state). Prints a table, the speed relative to a Tesla T4
// and the T4-Hours a test costs, and writes JSON. Lengths below 16K are not measured: their tests take seconds, and
// the time per iteration there is close to the 16K time. Works on any GPU hyprp runs on (CUDA or OpenCL).
// Unit of work: 1 T4-Hour = what one Tesla T4 card (running alone) computes in one hour. A test of n costs
// n * T4_REF[length][mode] / 3.6e6 T4-Hours; a GPU's speed in T4-Hours per hour is its "x T4" score.
// T4 reference ms/iter for 2^14 .. 2^21 words, modes 0..1: Kaggle Tesla T4 (CUDA backend, driver 580.178.04),
// mean of the two T4s of one session, each running alone, 2026-11-25 (hyprp -bench).
static const double T4_REF[8][2] = {{0.0226, 0.02275}, {0.0254, 0.0253}, {0.0345, 0.03495}, {0.0591, 0.0592},
                                    {0.1115, 0.1129},  {0.26245, 0.2671}, {0.6413, 0.64455}, {1.32285, 1.3281}};
static const char* const BENCH_MODE[2] = {"16-bit digits", "32-bit digits"};

static int runBench(int argc, char** argv) {
    int dev = 0;
    bool quick = false;
    std::string out;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-dev" && i + 1 < argc) dev = std::atoi(argv[++i]);
        else if (a == "-quick") quick = true;
        else if (a == "-out" && i + 1 < argc) out = argv[++i];
        else { logFatal("bad arg %s", a.c_str()); return 1; }
    }
    if (out.empty()) out = "hyprp_bench_dev" + std::to_string(dev) + ".json";
    const double scale = quick ? 0.25 : 1.0;
    const uint32_t K0 = 14, K1 = 21;
    const long long baseIters[8] = {24000, 18000, 12000, 9000, 6000, 3000, 1600, 800};
    struct Row { uint32_t k, mode; uint64_t L, n, nFrom, nTo; double ms[2]; double best; double maxerr; std::string opts; };
    std::vector<Row> rows;
    for (uint32_t k = K0; k <= K1; ++k) {
        const uint64_t L = 1ull << k;
        const uint64_t top = (uint64_t)((double)L * maxBitsFor(k));
        const uint64_t lo = (k > 6) ? (uint64_t)((double)(L / 2) * maxBitsFor(k - 1)) + 1 : 1;  // smallest n using L
        const uint64_t edges[3] = {lo, 16 * L + 1, top + 1};
        for (uint32_t mode = 0; mode < 2; ++mode) {
            const uint64_t from = std::max(edges[mode], lo), to = std::min(edges[mode + 1] - 1, top);
            if (from > to) continue;
            uint64_t n = mode == 1 ? to : from + (to - from) * 3 / 4;  // inside the range, away from its edges
            if (n % 2 == 0) --n;
            while (nearFraction(n, L) || n > to) n -= 2;
            rows.push_back(Row{k, mode, L, n, from, to, {0, 0}, 0, 0, ""});
        }
    }
    std::string device = "?";
    std::string exe = argv[0];
    for (int round = 0; round < 2; ++round) {
        for (auto& r : rows) {
            const long long it = std::max<long long>(200, (long long)(baseIters[r.k - K0] * scale)) / 4 * 4;
            char cmd[2048];
#ifdef _WIN32
            std::snprintf(cmd, sizeof(cmd), "\"\"%s\" %llu -L %u -iters %lld -report %lld -dev %d 2>&1\"", exe.c_str(),
                          (unsigned long long)r.n, r.k, it, it / 4, dev);
            FILE* f = _popen(cmd, "r");
#else
            std::snprintf(cmd, sizeof(cmd), "\"%s\" %llu -L %u -iters %lld -report %lld -dev %d 2>&1", exe.c_str(),
                          (unsigned long long)r.n, r.k, it, it / 4, dev);
            FILE* f = popen(cmd, "r");
#endif
            if (!f) { logFatal("cannot start %s", exe.c_str()); return 1; }
            char line[4096];
            double last = 0, err = 0;
            std::string log;
            while (std::fgets(line, sizeof(line), f)) {
                log += line;
                std::string s = logMessage(line);  // without the time stamp
                while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
                if (s.rfind("device:", 0) == 0) device = s.substr(8);
                if (s.rfind("options:", 0) == 0) r.opts = s.substr(9);
                double ms, e;
                long long a, b;
                if (std::sscanf(s.c_str(), "plain iter %lld/%lld %lf ms/iter maxerr %lf", &a, &b, &ms, &e) == 4) {
                    last = ms;
                    err = std::max(err, e);
                }
            }
#ifdef _WIN32
            _pclose(f);
#else
            pclose(f);
#endif
            if (last <= 0) {
                logFatal("FFT %lluK n=%llu failed:", (unsigned long long)(r.L >> 10), (unsigned long long)r.n);
                std::fputs(log.c_str(), stderr);
                return 1;
            }
            r.ms[round] = last;
            r.maxerr = std::max(r.maxerr, err);
            logInfo("round %d  FFT %5lluK  n=%-9llu %9.4f ms/iter  [%s]", round + 1, (unsigned long long)(r.L >> 10),
                    (unsigned long long)r.n, last, r.opts.c_str());
        }
    }
    for (auto& r : rows) r.best = std::min(r.ms[0], r.ms[1]);
    bool haveRef = true;
    for (auto& r : rows) haveRef = haveRef && T4_REF[r.k - K0][r.mode] > 0;
    logInfo("hyprp benchmark  device: %s", device.c_str());
    logInfo("  FFT    n range                 test n      ms/iter (2 rounds)   best     %s mode",
            haveRef ? " x T4  T4-Hours/test " : "");
    double lsum = 0;
    for (auto& r : rows) {
        char row[256], ref4[64] = "";
        if (haveRef) {
            const double ref = T4_REF[r.k - K0][r.mode], rel = ref / r.best;
            lsum += std::log(rel);
            std::snprintf(ref4, sizeof(ref4), "  %5.2f  %10.4f", rel, (double)r.n * ref / 3.6e6);
        }
        std::snprintf(row, sizeof(row), "  %5lluK %9llu..%-9llu  %9llu  %8.4f %8.4f  %8.4f%s   %s",
                      (unsigned long long)(r.L >> 10), (unsigned long long)r.nFrom, (unsigned long long)r.nTo,
                      (unsigned long long)r.n, r.ms[0], r.ms[1], r.best, ref4, BENCH_MODE[r.mode]);
        logInfo("%s", row);
    }
    const double score = haveRef ? std::exp(lsum / rows.size()) : 0;
    if (haveRef)
        logInfo("overall: %.3f x Tesla T4 (geometric mean over all lengths and modes) = %.3f T4-Hours per hour on this GPU",
                score, score);
    else logInfo("overall: T4 reference not set yet (these numbers become it when measured on a T4)");
    FILE* j = std::fopen(out.c_str(), "w");
    if (j) {
        std::string dv;
        for (char ch : device) { if (ch == '"' || ch == '\\') dv += '\\'; dv += ch; }
        std::fprintf(j, "{\n  \"program\": \"hyprp\",\n  \"c\": 25,\n  \"device\": \"%s\",\n  \"t4_relative\": %.4f,\n",
                     dv.c_str(), score);
        std::fprintf(j, "  \"modes\": [\"%s\", \"%s\"],\n", BENCH_MODE[0], BENCH_MODE[1]);
        std::fprintf(j, "  \"points\": [\n");
        for (size_t i = 0; i < rows.size(); ++i) {
            const Row& r = rows[i];
            const double ref = T4_REF[r.k - K0][r.mode];
            std::fprintf(j,
                         "    {\"fft\": %llu, \"mode\": %u, \"n_from\": %llu, \"n_to\": %llu, \"test_n\": %llu, "
                         "\"ms_per_iter\": %.5f, \"rounds\": [%.5f, %.5f], \"maxerr\": %.4f, \"t4_ms_per_iter\": %.5f}%s\n",
                         (unsigned long long)r.L, r.mode, (unsigned long long)r.nFrom, (unsigned long long)r.nTo,
                         (unsigned long long)r.n, r.best, r.ms[0], r.ms[1], r.maxerr, ref, i + 1 < rows.size() ? "," : "");
        }
        std::fprintf(j, "  ]\n}\n");
        std::fclose(j);
        logInfo("wrote %s", out.c_str());
    }
    return 0;
}

static const char* const VERSION = "1.0";
static const char* const DEVELOPED = "2026.11.25";

int main(int argc, char** argv) {
    logInfo("Starting dfprp v%s, Developed on %s", VERSION, DEVELOPED);
    logInfo("All Purpose k*b^n+c PRP tester on GPU");
    if (argc >= 2 && std::string(argv[1]) == "-bench") return runBench(argc, argv);
    if (argc < 2) {
        std::printf("usage: hyprp n [c=25] [-L log2L] [-iters k] [-report k] [-prof] [-B gec_block] [-nogec] [-fresh] [-dev i]\n"
                    "       hyprp -bench [-dev i] [-quick] [-out file.json]\n");
        return 1;
    }
    const uint32_t n = (uint32_t)std::atoi(argv[1]);
    int ai = 2;
    long long cc = 25;
    if (argc > 2 && argv[2][0] != '-') { cc = std::atoll(argv[2]); ai = 3; }
    int logL = 0;
    long long iters_limit = 0;
    bool prof = false;
    int devIdx = 0;
    long long report = 100000;
    int GB = 1000;  // Gerbicz block size (check every GB^2 iterations)
    bool gec = true, resume = true;
    for (; ai < argc; ++ai) {
        std::string a = argv[ai];
        if (a == "-L" && ai + 1 < argc) logL = std::atoi(argv[++ai]);
        else if (a == "-iters" && ai + 1 < argc) iters_limit = std::atoll(argv[++ai]);
        else if (a == "-prof") prof = true;
        else if (a == "-dev" && ai + 1 < argc) devIdx = std::atoi(argv[++ai]);
        else if (a == "-report" && ai + 1 < argc) report = std::atoll(argv[++ai]);
        else if (a == "-B" && ai + 1 < argc) GB = std::atoi(argv[++ai]);
        else if (a == "-nogec") gec = false;
        else if (a == "-fresh") resume = false;
        else { logFatal("bad arg %s", a.c_str()); return 1; }
    }
    // FFT length L = 2^k (M31 has only power-of-two roots here). Largest bits/word for each length, from round-off
    // surveys on Kaggle T4 (CUDA) and a Radeon 780M (OpenCL), 2-3 n per point, chosen so that the worst maxerr at every
    // cutoff is about the same (~0.27): 128K and shorter 17.7, 256K 17.45 (n <= 4,574,412), 512K 17.2 (9,017,753),
    // 1024K 16.95 (17,773,363), 2048K 16.5 (34,603,008); longer lengths were not measured, so larger n are refused.
    // n just below L * q / 4 (q integer) has an almost periodic digit-width pattern and ~0.05-0.15 more round-off at
    // the same bits/word, so within 32 of such a point the next length is used. -L k forces L = 2^k (the benchmark).
    uint64_t L = 0;
    if (logL) L = 1ull << logL;
    else
        for (uint64_t cand = 64;; cand *= 2) {
            if (cand > (1u << 21)) { L = cand; break; }  // refused below
            if ((double)n / (double)cand <= maxBitsFor(ilog2(cand))) {
                L = nearFraction(n, cand) && cand < (1u << 21) ? 2 * cand : cand;
                break;
            }
        }
    if (L > (1u << 21)) {  // longer lengths have not been measured
        logFatal("n = %u is too large: hyprp handles n up to 34,603,008 (FFT length 2048K)", n);
        return 2;
    }
    // four-step layout: W x H complex elements (W = 512 lines, wider when H would pass 1024), TG tail columns per
    // work-group, WG work-items; W=512 TG=2 WG=128 was the fastest on Kaggle T4
    const uint64_t NC = L / 2;
    uint32_t W = 512;
    while (W > 2 && (NC % W != 0 || NC / W < 2)) W >>= 1;
    while (NC / W > 1024 && NC % (2 * W) == 0) W *= 2;  // tail LDS: 32*TG*H bytes
    const uint32_t H = (uint32_t)(NC / W);
    uint32_t TGc = 2;
    while (TGc > 1 && (2 * TGc * H > 2048 || (W / 2) % TGc)) TGc >>= 1;  // tail LDS = 32 * TG * H bytes
    const uint32_t WGc = 128;
    logInfo("n=%u c=%lld  L=%llu (%.3f bits/word)  W=%u H=%u TG=%u WG=%u  [FP32+M31]", n, cc, (unsigned long long)L,
            (double)n / L, W, H, TGc, WGc);
    // radix-8/4/2 Stockham stages of a power-of-two line FFT, as STW(R, PS) / STH(R, PS) items for hyprp.cl
    auto stages = [](uint32_t P, const char* tag) {
        std::string out;
        uint32_t ps = 1, rest = P;
        auto add = [&](uint32_t R) { char b[64]; std::snprintf(b, sizeof(b), "%s(%u,%u) ", tag, R, ps); out += b; ps *= R; rest /= R; };
        uint32_t e = 0;
        while ((1u << e) < rest) ++e;
        uint32_t n8 = e / 3, n4 = 0, n2 = 0;
        if (e % 3 == 2) n4 = 1;
        else if (e % 3 == 1) { if (n8) { --n8; n4 = 2; } else n2 = 1; }  // prefer 8^k*4*4 over 8^k*2
        while (n8--) add(8);
        while (n4--) add(4);
        while (n2--) add(2);
        if (rest != 1) { logFatal("unsupported length %u", P); std::exit(1); }
        return out;
    };

    // ---- GPU setup: CUDA on NVIDIA (when built with HYPRP_CUDA), OpenCL otherwise
    std::unique_ptr<Gpu> gpu = makeGpu(devIdx, prof);
    std::string exe_dir = argv[0];
    auto pos = exe_dir.find_last_of("\\/");
    exe_dir = pos == std::string::npos ? "" : exe_dir.substr(0, pos + 1);
    std::ifstream kf(exe_dir + "hyprp.cl");
    if (!kf) { logFatal("cannot open hyprp.cl"); return 1; }
    std::stringstream ss;
    ss << kf.rdbuf();
    const uint32_t NH = (uint32_t)(L / 512 + 2);
    // digits are stored as 16-bit pairs when every (balanced) digit fits in int16, i.e. n <= 16 L
    const bool dig16 = (n + L - 1) / L <= 16;
    const size_t DB = dig16 ? 4 : 8;  // bytes per element of the digit array
    logInfo("options: digits %s", dig16 ? "16-bit" : "32-bit");
    uint32_t MUc = 0;
    for (uint32_t u = 1; u < 31; ++u) if ((L % 31) * u % 31 == 1) MUc = u;
    {
        char b[512];
        std::snprintf(b, sizeof(b),
                      "#define W %uu\n#define H %uu\n#define NEXP %uu\n#define CC %lldL\n#define TG %uu\n#define NH %uu\n"
                      "#define WG %u\n#define MU %uu\n#define LOGN %uu\n",
                      W, H, n, cc, TGc, NH, WGc, MUc, ilog2(NC));
        std::string src = b;
        if (dig16) src += "#define DIG16 1\n";
        src += "#define WSTAGES " + stages(W, "STW") + "\n";
        src += "#define HSTAGES " + stages(H, "STH") + "\n";
        src += ss.str();
        std::string log;
        if (!gpu->build(src, log)) {
            if (gpu->backend == "OpenCL") {
                logFatal("kernel build failed (OpenCL):");
                std::fputs(log.c_str(), stderr);
                return 1;
            }
            // e.g. a GPU the installed CUDA toolkit no longer compiles for: the OpenCL driver may still run it
            logWarn("kernel build failed (%s), falling back to OpenCL:", gpu->backend.c_str());
            std::fputs(log.c_str(), stdout);
            gpu.reset(new GpuCL(devIdx, prof));
            log.clear();
            if (!gpu->build(src, log)) {
                logFatal("kernel build failed (OpenCL):");
                std::fputs(log.c_str(), stderr);
                return 1;
            }
        }
    }
    GpuKernel kF = gpu->kernel("fwdW"), kT = gpu->kernel("tail"), kTF = gpu->kernel("tailFwd");
    GpuKernel kTM = gpu->kernel("tailMul"), kI = gpu->kernel("invW");

    // ---- tables: twiddles as (FP32 complex, GF(p^2) element); weights as (FP32 factor, GF(p) factor)
    if (mpow((uint64_t)cc % P31, (P31 - 1) / 2) != 1) {
        logFatal("c = %lld is not a square mod 2^31-1: its L-th root does not exist", cc);
        return 1;
    }
    const uint32_t KN = ilog2(NC);
    const G2 gN = m31root(KN);
    struct hc { float a, b; uint32_t c, d; };
    auto toh = [](cplx v, G2 m) { return hc{(float)v.x, (float)v.y, (uint32_t)m.re, (uint32_t)m.im}; };
    std::vector<hc> twW(W), twH(H), twN1(W), twN2(H);
    for (uint32_t i = 0; i < W; ++i) {
        twW[i] = toh(root(i, W), gpow(gN, (NC / W) * (uint64_t)i));
        twN1[i] = toh(root(i, NC), gpow(gN, i));
    }
    for (uint32_t i = 0; i < H; ++i) {
        twH[i] = toh(root(i, H), gpow(gN, (NC / H) * (uint64_t)i));
        twN2[i] = toh(root((uint64_t)W * i, NC), gpow(gN, (uint64_t)W * i));
    }
    const uint32_t WSn = 512;
    std::vector<float> WTAB(2 * (2 * NH + 2 * WSn), 0.f);
    std::vector<uint32_t> ITAB(2 * NH + 2 * WSn, 0);
    {
        const double log2c = std::log2((double)cc);
        const int64_t half = (int64_t)(P31 - 1) / 2;  // odd; the order of c divides it
        const uint64_t kappa = mpow((uint64_t)cc, (uint64_t)invmod((int64_t)(L % half), half));  // c^(1/L)
        if (mpow(kappa, L) != (uint64_t)cc % P31) { logFatal("c^(1/L) failed"); return 1; }
        const uint64_t ikappa = mpow(kappa, P31 - 2);
        auto put = [&](size_t idx, double v, uint64_t m) {
            WTAB[2 * idx] = (float)v;
            uint32_t mb = (uint32_t)m;
            std::memcpy(&WTAB[2 * idx + 1], &mb, 4);
        };
        for (uint32_t jh = 0; jh < NH; ++jh) {
            uint64_t x = (uint64_t)n * jh * WSn;
            uint64_t r = x % L, f = x / L;
            ITAB[jh] = (uint32_t)r;
            ITAB[NH + WSn + jh] = (uint32_t)f;
            double wv = std::exp2(-(double)r / (double)L + (double)jh * WSn * log2c / (double)L);
            put(jh, wv, mpow(kappa, (uint64_t)jh * WSn));
            put(NH + WSn + jh, 1.0 / (wv * (double)NC), mpow(ikappa, (uint64_t)jh * WSn));
        }
        for (uint32_t jl = 0; jl < WSn; ++jl) {
            uint64_t x = (uint64_t)n * jl;
            uint64_t r = x % L, f = x / L;
            ITAB[NH + jl] = (uint32_t)r;
            ITAB[2 * NH + WSn + jl] = (uint32_t)f;
            double wv = std::exp2(-(double)r / (double)L + (double)jl * log2c / (double)L);
            put(NH + jl, wv, mpow(kappa, jl));
            put(2 * NH + WSn + jl, 1.0 / wv, mpow(ikappa, jl));
        }
    }

    auto mk = [&](size_t bytes, const void* init) { return gpu->alloc(bytes, init); };
    GpuBuf bTwW = mk(16 * W, twW.data()), bTwH = mk(16 * H, twH.data());
    GpuBuf bTwN1 = mk(16 * W, twN1.data()), bTwN2 = mk(16 * H, twN2.data());
    GpuBuf bWT = mk(4 * WTAB.size(), WTAB.data()), bIWT = bWT;
    GpuBuf bIT = mk(4 * ITAB.size(), ITAB.data());
    GpuBuf bA = mk(16 * NC, nullptr), bA2 = mk(16 * NC, nullptr);
    uint32_t zero = 0;
    GpuBuf bErr = mk(4, &zero);

    struct Num { GpuBuf D, CY; };
    std::vector<int32_t> Dh(2 * NC, 0);
    std::vector<int64_t> CYh(2 * NC, 0);  // host copy of the carries: (low, high) per element, always 64-bit
    // on the GPU each element's carry is one 64-bit word: lo in bits 24..63, q2 in bits 0..23
    std::vector<int64_t> CYp(NC);
    auto readCY = [&](GpuBuf b) {
        gpu->read(b, 8 * NC, CYp.data());
        for (size_t i = 0; i < NC; ++i) {
            CYh[2 * i] = CYp[i] >> 24;
            CYh[2 * i + 1] = (int64_t)((uint64_t)CYp[i] << 40) >> 40;
        }
    };
    auto writeCY = [&](GpuBuf b, const int64_t* src) {
        for (size_t i = 0; i < NC; ++i)
            CYp[i] = (int64_t)(((uint64_t)src[2 * i] << 24) | ((uint64_t)src[2 * i + 1] & 0xFFFFFFull));
        gpu->write(b, 0, 8 * NC, CYp.data());
    };
    // digits on the GPU are int32 or int16 pairs (DB bytes per element); the host always keeps int32 (Dh)
    std::vector<int16_t> D16(dig16 ? 2 * NC : 0);
    auto readD = [&](GpuBuf b) {
        if (!dig16) { gpu->read(b, 8 * NC, Dh.data()); return; }
        gpu->read(b, 4 * NC, D16.data());
        for (size_t i = 0; i < 2 * NC; ++i) Dh[i] = D16[i];
    };
    auto writeD = [&](GpuBuf b, const int32_t* src) {
        if (!dig16) { gpu->write(b, 0, 8 * NC, src); return; }
        for (size_t i = 0; i < 2 * NC; ++i) D16[i] = (int16_t)src[i];
        gpu->write(b, 0, 4 * NC, D16.data());
    };
    auto mkNum = [&]() { return Num{mk(DB * NC, nullptr), mk(8 * NC, nullptr)}; };
    auto setSmall = [&](Num a, int v) {  // a = v (small)
        std::vector<int32_t> d(2 * NC, 0);
        std::vector<int64_t> cy(2 * NC, 0);
        d[0] = v;
        writeD(a.D, d.data());
        writeCY(a.CY, cy.data());
    };
    auto copyNum = [&](Num dst, Num src) {
        gpu->copy(dst.D, src.D, DB * NC);
        gpu->copy(dst.CY, src.CY, 8 * NC);
    };
    Num X = mkNum(), Dg = mkNum(), Xs = mkNum(), Tg = mkNum();

    for (GpuKernel k : {kT, kTF, kTM}) {
        gpu->setArg(k, 1, bTwH);
        gpu->setArg(k, 2, bTwN1);
        gpu->setArg(k, 3, bTwN2);
    }
    gpu->setArg(kT, 0, bA);
    gpu->setArg(kTF, 0, bA2);
    gpu->setArg(kTM, 0, bA);
    gpu->setArg(kTM, 4, bA2);
    gpu->setArg(kF, 2, bWT);
    gpu->setArg(kF, 4, bTwW);
    gpu->setArg(kF, 5, bTwN1);
    gpu->setArg(kF, 6, bTwN2);
    gpu->setArg(kI, 0, bA);
    gpu->setArg(kI, 3, bIWT);
    gpu->setArg(kI, 4, bTwW);
    gpu->setArg(kI, 5, bTwN1);
    gpu->setArg(kI, 6, bTwN2);
    gpu->setArg(kI, 8, bErr);
    gpu->setArg(kI, 9, bIT);
    gpu->setArg(kF, 7, bIT);

    const size_t WGS = WGc;
    const size_t gF = H, gT = W / (2 * TGc), gI = H;  // work-groups per kernel
    GpuBuf curF, curFout, curI;
    int curMul = -1;
    bool profNow = false;
    double tF = 0, tT = 0, tI = 0;
    long long nprof = 0;
    auto enq = [&](GpuKernel k, size_t groups, double* acc) { gpu->launch(k, groups, WGS, profNow ? acc : nullptr); };
    auto fwd = [&](Num a, GpuBuf out) {
        if (!(curF == a.D)) {
            gpu->setArg(kF, 0, a.D);
            gpu->setArg(kF, 1, a.CY);
            curF = a.D;
        }
        if (!(curFout == out)) { gpu->setArg(kF, 3, out); curFout = out; }
        enq(kF, gF, &tF);
    };
    auto inv = [&](Num a, int mulc) {
        if (!(curI == a.D)) {
            gpu->setArg(kI, 1, a.D);
            gpu->setArg(kI, 2, a.CY);
            curI = a.D;
        }
        if (mulc != curMul) { gpu->setArg(kI, 7, (int32_t)mulc); curMul = mulc; }
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
    auto throttle = [&]() { gpu->throttle(); };

    // ---- canonical residue (host): X = sum d_j 2^B_j + sum CY_m 2^B_(2m+2), reduced into [0, N)
    const size_t K = (n + 31) / 32 + 1;
    auto Bj = [&](uint64_t j) { return ((uint64_t)n * j + L - 1) / L; };
    auto canon = [&](Num a) {
        readD(a.D);
        readCY(a.CY);
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
                addAt(Bj(2 * m + 2), CYh[2 * p]);
                addAt(Bj(2 * m + 4), CYh[2 * p + 1]);
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
    bool inGec = false, suspect = false;  // round-off above 0.42 inside a Gerbicz block forces that block to be redone
    auto progress = [&](long long done, const char* tag) {
        uint32_t e2 = 0;
        gpu->read(bErr, 4, &e2);
        gpu->write(bErr, 0, 4, &zero);
        auto now = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(now - t0).count();
        double seg = std::chrono::duration<double>(now - tl).count();
        double msit = done > itl ? 1000.0 * seg / (done - itl) : 0;
        logInfo("%s iter %lld/%lld  %.4f ms/iter  maxerr %.4f  ETA %.1f min  (elapsed %.1f s)", tag, done, todo, msit,
                e2 / 4294967295.0, msit * (todo - done) / 60000.0, el);
        tl = now; itl = done;
        if (e2 / 4294967295.0 > 0.42) {
            if (inGec) {  // a digit may have been rounded the wrong way: redo the block from the last Gerbicz check
                logWarn("round-off %.4f too large: the current Gerbicz block will be redone", e2 / 4294967295.0);
                suspect = true;
            } else {
                logFatal("round-off too large");
                std::exit(3);
            }
        }
    };
    auto printRes = [&](const std::vector<uint32_t>& limb, bool full) {
        bool isone = limb[0] == 1;
        for (size_t i = 1; i < K && isone; ++i) isone = limb[i] == 0;
        unsigned long long res64 = (unsigned long long)limb[0] | ((unsigned long long)limb[1] << 32);
        logInfo("RES64 %016llX", res64);
        std::string r2048;
        for (int i = 63; i >= 0; --i) {
            char b[9];
            std::snprintf(b, sizeof(b), "%08X", i < (int)K ? limb[i] : 0u);
            r2048 += b;
        }
        logInfo("RES2048 %s", r2048.c_str());
        if (full) logInfo("2^%u-%lld is %s", n, cc, isone ? "a probable prime!" : "composite");
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
        gpu->finish();
        gpu->flushProf();
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        logInfo("done %lld iterations in %.2f s  (%.4f ms/iter)", todo, el, 1000.0 * el / todo);
        if (nprof)
            logInfo("kernel ms (avg of %lld): fwdW %.4f  tail %.4f  invW %.4f  sum %.4f", nprof, tF / nprof, tT / nprof,
                    tI / nprof, (tF + tT + tI) / nprof);
        printRes(canon(X), todo == total);
        return 0;
    }

    // ---- full test with Gerbicz check.
    // Iterations 0 .. G-1 all have mulc = 3 (exponent bits >= 40). y = 3x then satisfies y -> y^2,
    // so with d_s = prod_{u<=s} y_u (one y every B iterations): d_s == y_0 * d_(s-1)^(2^B).
    const long long G = (long long)n - 41;
    char ckname[128];
    std::snprintf(ckname, sizeof(ckname), "hyprp_%u_%lld.ckpt", n, cc);
    long long it = 0;
    auto saveCk = [&](long long iter) {
        readD(X.D);
        readCY(X.CY);
        std::string tmp = std::string(ckname) + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) return;
        uint64_t hdr[4] = {n, (uint64_t)cc, (uint64_t)L, (uint64_t)iter};
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
                hdr[2] == (uint64_t)L && std::fread(Dh.data(), 4, Dh.size(), f) == Dh.size() &&
                std::fread(CYh.data(), 8, CYh.size(), f) == CYh.size()) {
                writeD(X.D, Dh.data());
                writeCY(X.CY, CYh.data());
                it = (long long)hdr[3];
                logInfo("resumed from %s at iteration %lld", ckname, it);
            }
            std::fclose(f);
        }
    }
    itl = itStart = it;
    int fails = 0;
    const long long S = (long long)GB * GB;
    inGec = true;
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
                if ((j + 1) % report == 0) progress(j + 1, "gec");
            }
            if (b == blocks - 1) copyNum(Tg, Dg);  // d_(s-1)
            mul(Dg, X, 3);                          // d_s = d_(s-1) * y_s
            throttle();
        }
        for (int k = 0; k < GB; ++k) { sqr(Tg, 1); throttle(); }
        mul(Tg, Xs, 3);
        gpu->finish();
        {  // round-off since the last progress line (the reset makes the next progress line cover less)
            uint32_t e2 = 0;
            gpu->read(bErr, 4, &e2);
            gpu->write(bErr, 0, 4, &zero);
            if (e2 / 4294967295.0 > 0.42) {
                logWarn("round-off %.4f too large before the Gerbicz check: block redone", e2 / 4294967295.0);
                suspect = true;
            }
        }
        bool ok = !suspect && canon(Tg) == canon(Dg);
        suspect = false;
        if (ok) {
            it = j;
            saveCk(it);
            logInfo("Gerbicz check OK at iteration %lld", it);
        } else {
            ++fails;
            logWarn("Gerbicz check FAILED at iteration %lld (fail #%d), rolling back to %lld", j, fails, it);
            if (fails >= 5) { logFatal("too many errors"); return 4; }
            copyNum(X, Xs);
        }
    }
    // remaining iterations (< B not covered by GEC, plus the 40 low bits): run twice and compare
    inGec = false;
    std::vector<uint32_t> r1, r2;
    copyNum(Xs, X);
    bool agreed = false;
    for (int attempt = 0; attempt < 6 && !agreed; ++attempt) {
        if (attempt) copyNum(X, Xs);
        for (long long j = it; j < total; ++j) { sqr(X, mulcOf(j)); throttle(); }
        gpu->finish();
        auto r = canon(X);
        if ((!r1.empty() && r == r1) || (!r2.empty() && r == r2)) { r1 = r; agreed = true; }
        else if (r1.empty()) r1 = r;
        else r2 = r;
    }
    if (!agreed) { logFatal("final iterations not reproducible"); return 5; }
    progress(total, "final");
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    logInfo("done %lld iterations in %.2f s (%.4f ms/iter incl. checks)", total - itStart, el,
            1000.0 * el / std::max<long long>(1, total - itStart));
    printRes(r1, true);
    std::remove(ckname);
    return 0;
}
