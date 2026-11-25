// hyprp: weighted real FFT mod 2^n - c (four-step W x H: fwdW, tail, invW), computed twice at once: in FP32
// (approximate) and exactly in GF(p^2), p = M31 = 2^31 - 1 (a "complex" NTT; i^2 = -1).
// Every complex value is a float4: .x/.y = FP32 complex, .z/.w = the GF(p^2) element (bit pattern of two uints).
// The exact result z_j of the convolution is rebuilt as z = r + p * round((z_fp32 - r) / p) with r = z mod p
// from the NTT, so the FP32 part only has to be accurate to +-p/2 and words can be ~15-17 bits wide.
// GF(p) facts used: 2^31 == 1, so multiplying by a power of two is a 31-bit rotation; 1/2 = 2^30, 1/sqrt2 = 2^15;
// 2 and c = 25 have odd order, so the DWT weights 2^(k/L) and c^(j/L) exist in GF(p).
// The same source is compiled as OpenCL C or, with HYPRP_CUDA defined, as CUDA C++ (NVRTC); the macros below
// hide the differences, so the kernel code uses only scalar components and these helpers.
// Compile-time constants: W, H, NEXP, CC, TG, NH, WG, WSTAGES, HSTAGES, MU (= L^-1 mod 31), LOGN (= log2 N).
// DIG16 (set by the host when no digit is wider than 16 bits): digits are stored as 16-bit pairs.
// Carries: the carry out of element m is c1 ~ bits/word + log2(c) + 13 bits. It is split into lo (|lo| < 2^(e-1),
// e = width of digits 2m+2 and 2m+3, added at digit 2m+2 of element m+1) and q2 = round(c1 / 2^e) (added at digit
// 2m+4 of element m+2, almost always 0). Both are packed into one 64-bit word per element, lo in the upper 40 bits
// and q2 in the low 24 bits, so the carry array costs 8 bytes per element whatever the word size.

#ifdef HYPRP_CUDA
typedef unsigned int uint;
typedef unsigned long long ulong;  // long is 64-bit on the LP64 Linux targets
#define KERNEL extern "C" __global__ void __launch_bounds__(WG)
#define GLOBAL
#define LOCAL
#define LDS __shared__
#define DEVFN static __device__ __forceinline__
#define BARRIER() __syncthreads()
DEVFN uint group_id() { return blockIdx.x; }
DEVFN uint local_id() { return threadIdx.x; }
DEVFN float4 F4(float a, float b, float c, float d) { return make_float4(a, b, c, d); }
DEVFN int2 I2(int a, int b) { return make_int2(a, b); }
#ifdef DIG16
typedef short2 DT;
DEVFN DT MKD(int a, int b) { return make_short2((short)a, (short)b); }
#else
typedef int2 DT;
DEVFN DT MKD(int a, int b) { return make_int2(a, b); }
#endif
typedef long long CS;  // packed carry
DEVFN uint F2U(float f) { return __float_as_uint(f); }
DEVFN float U2F(uint u) { return __uint_as_float(u); }
#define FMAF fmaf
#define RINTF rintf
#define FABSF fabsf
#define FMAXF fmaxf
#define LDEXPF ldexpf
DEVFN uint atom_max(uint* p, uint v) { return atomicMax(p, v); }
#else
#define KERNEL __kernel __attribute__((reqd_work_group_size(WG, 1, 1))) void
#define GLOBAL __global
#define LOCAL __local
#define LDS __local
#define DEVFN inline __attribute__((always_inline))
#define BARRIER() barrier(CLK_LOCAL_MEM_FENCE)
#define group_id() ((uint)get_group_id(0))
#define local_id() ((uint)get_local_id(0))
#define F4(a, b, c, d) ((float4)((a), (b), (c), (d)))
#define I2(a, b) ((int2)((a), (b)))
#ifdef DIG16
typedef short2 DT;
#define MKD(a, b) ((short2)((short)(a), (short)(b)))
#else
typedef int2 DT;
#define MKD(a, b) ((int2)((a), (b)))
#endif
typedef long CS;  // packed carry
#define F2U(f) as_uint(f)
#define U2F(u) as_float(u)
#define FMAF fma
#define RINTF rint
#define FABSF fabs
#define FMAXF fmax
#define LDEXPF ldexp
#define atom_max atomic_max
#endif

typedef float2 df;     // weight table entry: (FP32 weight, bits of the GF(p) weight factor)
typedef float4 C;
#define NC (W * H)
#define LL (2 * NC)
#ifndef WG
#define WG 64
#endif
#define WS 512u  // weight table split: j = jh * WS + jl
#define MP 0x7FFFFFFFu
#define SQRT1_2F 0.70710678118654752440f

// ---- GF(p) arithmetic on values in [0, p] (p itself also stands for 0)
DEVFN uint mred(uint x) { return (x & MP) + (x >> 31); }
DEVFN uint madd(uint a, uint b) { return mred(a + b); }
DEVFN uint msub(uint a, uint b) { return mred(a + (MP - b)); }
DEVFN uint mneg(uint a) { return MP - a; }
DEVFN uint mmul(uint a, uint b) {
    ulong t = (ulong)a * b;
    return mred((uint)(t & MP) + (uint)(t >> 31));
}
DEVFN uint mrot(uint x, uint s) { return s ? (((x << s) & MP) | (x >> (31 - s))) : x; }  // x * 2^s, s < 31
DEVFN uint mhalf(uint x) { return (x >> 1) | ((x & 1u) << 30); }                     // x / 2

// ---- packed carries: lo in bits 24..63, q2 in bits 0..23 (two's complement)
DEVFN CS packCY(long lo, long q2) { return (CS)(((ulong)lo << 24) | ((ulong)q2 & 0xFFFFFFul)); }
DEVFN long cyLo(CS v) { return (long)(v >> 24); }
DEVFN long cyHi(CS v) { return (long)((CS)((ulong)v << 40) >> 40); }

// ---- hybrid complex values
DEVFN C MKH(float fx, float fy, uint mx, uint my) { return F4(fx, fy, U2F(mx), U2F(my)); }
DEVFN uint MRE(C a) { return F2U(a.z); }
DEVFN uint MIM(C a) { return F2U(a.w); }
DEVFN C cadd(C a, C b) { return MKH(a.x + b.x, a.y + b.y, madd(MRE(a), MRE(b)), madd(MIM(a), MIM(b))); }
DEVFN C csub(C a, C b) { return MKH(a.x - b.x, a.y - b.y, msub(MRE(a), MRE(b)), msub(MIM(a), MIM(b))); }
DEVFN C cmul(C a, C b) {
    const uint ar = MRE(a), ai = MIM(a), br = MRE(b), bi = MIM(b);
    return MKH(FMAF(a.x, b.x, -a.y * b.y), FMAF(a.x, b.y, a.y * b.x),
               msub(mmul(ar, br), mmul(ai, bi)), madd(mmul(ar, bi), mmul(ai, br)));
}
DEVFN C conjg(C a) { return MKH(a.x, -a.y, MRE(a), mneg(MIM(a))); }
// multiply by -i (forward) or +i (inverse)
DEVFN C mulmi(C a, int dir) {
    return dir > 0 ? MKH(a.y, -a.x, MIM(a), mneg(MRE(a))) : MKH(-a.y, a.x, mneg(MIM(a)), MRE(a));
}
DEVFN C muli(C a) { return MKH(-a.y, a.x, mneg(MIM(a)), MRE(a)); }
DEVFN C chalf(C a) { return MKH(a.x * 0.5f, a.y * 0.5f, mhalf(MRE(a)), mhalf(MIM(a))); }
DEVFN C csqrthalf(C a) { return MKH(a.x * SQRT1_2F, a.y * SQRT1_2F, mrot(MRE(a), 15), mrot(MIM(a), 15)); }

DEVFN void dft2(C* x) { C t = x[0]; x[0] = cadd(t, x[1]); x[1] = csub(t, x[1]); }
DEVFN void dft4(C* x, int dir) {
    C v0 = cadd(x[0], x[2]), v1 = csub(x[0], x[2]), v2 = cadd(x[1], x[3]), v3 = mulmi(csub(x[1], x[3]), dir);
    x[0] = cadd(v0, v2); x[2] = csub(v0, v2); x[1] = cadd(v1, v3); x[3] = csub(v1, v3);
}
// the GF(p^2) twiddle tables are built so that w_P^(P/4) = -i and w_P^(P/8) = (1 - i)/sqrt2, as in C
DEVFN void dft8(C* x, int dir) {
    C e[4], o[4];
    for (int q = 0; q < 4; ++q) { e[q] = x[2 * q]; o[q] = x[2 * q + 1]; }
    dft4(e, dir); dft4(o, dir);
    C a = o[1], b = o[3];
    o[1] = csqrthalf(cadd(a, mulmi(a, dir)));
    o[3] = csqrthalf(csub(mulmi(b, dir), b));
    o[2] = mulmi(o[2], dir);
    for (int q = 0; q < 4; ++q) { x[q] = cadd(e[q], o[q]); x[q + 4] = csub(e[q], o[q]); }
}

// One Stockham stage, in place in local memory a (NB arrays of length P), radix R (8, 4 or 2), sub-length PS.
// tw[j] = exp(-2 pi i j / P).
#define FFT_STAGE(a, P, NB, R, PS, tw, dir)                                                          \
    {                                                                                                \
        const uint nbf = (P) / (R), tot = (NB) * nbf;                                                \
        C x[((NB) * ((P) / (R)) + WG - 1) / WG][8];                                                  \
        _Pragma("unroll") for (uint u = 0; u < ((NB) * ((P) / (R)) + WG - 1) / WG; ++u) {           \
            const uint g = t + u * WG;                                                               \
            if (g < tot) {                                                                           \
                const uint arr = g / nbf, i = g % nbf;                                               \
                _Pragma("unroll") for (uint r = 0; r < (R); ++r) x[u][r] = a[arr * (P) + i + r * nbf]; \
            }                                                                                        \
        }                                                                                            \
        BARRIER();                                                                                   \
        _Pragma("unroll") for (uint u = 0; u < ((NB) * ((P) / (R)) + WG - 1) / WG; ++u) {           \
            const uint g = t + u * WG;                                                               \
            if (g < tot) {                                                                           \
                const uint arr = g / nbf, i = g % nbf, k = i % (PS);                                 \
                if ((PS) > 1) {                                                                      \
                    const uint stepi = k * ((P) / ((R) * (PS)));                                     \
                    _Pragma("unroll") for (uint r = 1; r < (R); ++r) {                               \
                        C w = tw[r * stepi];                                                         \
                        if ((dir) < 0) w = conjg(w);                                                 \
                        x[u][r] = cmul(x[u][r], w);                                                  \
                    }                                                                                \
                }                                                                                    \
                if ((R) == 8) dft8(x[u], dir);                                                       \
                else if ((R) == 4) dft4(x[u], dir);                                                  \
                else dft2(x[u]);                                                                     \
                const uint j = (i - k) * (R) + k;                                                    \
                _Pragma("unroll") for (uint r = 0; r < (R); ++r) a[arr * (P) + j + r * (PS)] = x[u][r]; \
            }                                                                                        \
        }                                                                                            \
        BARRIER();                                                                                   \
    }
// WSTAGES / HSTAGES expand to a list of STW(R, PS) / STH(R, PS) items (host generated). NB_ must be #defined
// to a literal around each use.
#define STW(R, PS) FFT_STAGE(a_, W, NB_, R, PS, tw_, dir_)
#define STH(R, PS) FFT_STAGE(a_, H, NB_, R, PS, tw_, dir_)
#define FFTW_RUN(a, tw, dir) { LOCAL C* a_ = (a); GLOBAL const C* tw_ = (tw); const int dir_ = (dir); WSTAGES }
#define FFTH_RUN(a, tw, dir) { LOCAL C* a_ = (a); GLOBAL const C* tw_ = (tw); const int dir_ = (dir); HSTAGES }

// ---- digit positions and weights.  B(j) = ceil(n j / L) = FH + FL + ceil((RH + RL) / L)
// itab = [RH(NH) | RL(WS) | FH(NH) | FL(WS)]
// wtab = [WH(NH) | WL(WS) | iWH(NH) | iWL(WS)], each entry (FP32 factor, bits of the GF(p) factor c^(.../L))
DEVFN uint Bpos2(GLOBAL const uint* it, uint j, uint* ce, uint* rs) {
    const uint jh = j / WS, jl = j % WS;
    *rs = it[jh] + it[NH + jl];
    *ce = *rs == 0 ? 0u : (*rs <= LL ? 1u : 2u);
    return it[NH + WS + jh] + it[2 * NH + WS + jl] + *ce;
}
DEVFN uint Bpos(GLOBAL const uint* it, uint j, uint* ce) { uint rs; return Bpos2(it, j, ce, &rs); }
// GF(p) weight 2^((ce L - rs)/L) = 2^(MU (ce L - rs) mod 31): a rotation
DEVFN uint wrot(uint ce, uint rs) {  // ce <= 2, rs <= 2 LL: everything fits in 32 bits
    const uint v = (ce * (LL % 31u) + 62u - rs % 31u) % 31u;
    return (v * MU) % 31u;
}
DEVFN float wgtF(GLOBAL const df* wt, uint j, uint ce) { return LDEXPF(wt[j / WS].x * wt[NH + j % WS].x, (int)ce); }
DEVFN float iwgtF(GLOBAL const df* wt, uint j, uint ce) {
    return LDEXPF(wt[NH + WS + j / WS].x * wt[2 * NH + WS + j % WS].x, -(int)ce);
}
DEVFN uint wgtM(GLOBAL const df* wt, uint j, uint ce, uint rs) {
    return mrot(mmul(F2U(wt[j / WS].y), F2U(wt[NH + j % WS].y)), wrot(ce, rs));
}
DEVFN uint iwgtM(GLOBAL const df* wt, uint j, uint ce, uint rs) {  // includes 1/N = 2^-LOGN
    return mrot(mmul(F2U(wt[NH + WS + j / WS].y), F2U(wt[2 * NH + WS + j % WS].y)),
                (62 - wrot(ce, rs) - (LOGN % 31)) % 31);
}

DEVFN long bal(long v, uint b, long* carry) {
    const long base = 1L << b, hb = base >> 1;
    long d = ((v + hb) & (base - 1)) - hb;
    *carry = (v - d) >> b;
    return d;
}
DEVFN uint tomod(long d) { return d < 0 ? (uint)(d + MP) : (uint)d; }  // |d| < p

// twN1[i] = w_N^i for i < W ; twN2[i] = w_N^(W i) = w_H^i for i < H
DEVFN C twN(GLOBAL const C* twN1, GLOBAL const C* twN2, uint e) { return cmul(twN1[e % W], twN2[e / W]); }
#define TWN(e) twN(twN1, twN2, (e))

// digits (d0, d1) of complex element m -> weighted hybrid value
DEVFN C weighed(GLOBAL const df* WT, GLOBAL const uint* IT, uint m, long d0, long d1) {
    uint ce0, ce1, rs0, rs1;
    Bpos2(IT, 2 * m, &ce0, &rs0);
    Bpos2(IT, 2 * m + 1, &ce1, &rs1);
    return MKH((float)d0 * wgtF(WT, 2 * m, ce0), (float)d1 * wgtF(WT, 2 * m + 1, ce1),
               mmul(tomod(d0), wgtM(WT, 2 * m, ce0, rs0)), mmul(tomod(d1), wgtM(WT, 2 * m + 1, ce1, rs1)));
}

KERNEL fwdW(GLOBAL const DT* D, GLOBAL const CS* CY, GLOBAL const df* WT, GLOBAL C* A,
            GLOBAL const C* twW, GLOBAL const C* twN1, GLOBAL const C* twN2, GLOBAL const uint* IT) {
    LDS C s1[W];
    const uint h = group_id(), t = local_id();
    for (uint w = t; w < W; w += WG) {
        const uint pos = h * W + w;
        const uint m = h + H * w;
        // two-level carry: lo of element m-1 and q2 of element m-2 (both land at digit 2m)
        const uint m1 = m == 0 ? NC - 1 : m - 1;
        const uint m2 = m < 2 ? m + NC - 2 : m - 2;
        const long cin = cyLo(CY[(m1 % H) * W + m1 / H]) * (m == 0 ? CC : 1) +
                         cyHi(CY[(m2 % H) * W + m2 / H]) * (m < 2 ? CC : 1);
        const DT d = D[pos];
        uint ce0, B1 = Bpos(IT, 2 * m + 1, &ce0), B0 = Bpos(IT, 2 * m, &ce0);
        long c0;
        long d0 = bal((long)d.x + cin, B1 - B0, &c0);
        long d1 = (long)d.y + c0;
        s1[w] = weighed(WT, IT, m, d0, d1);
    }
    BARRIER();
#define NB_ 1
    FFTW_RUN(s1, twW, 1)
#undef NB_
    for (uint kw = t; kw < W; kw += WG) A[h * W + kw] = cmul(s1[kw], TWN(h * kw));
}

// square step for a pair (k, N-k): Vk, Vn in/out. t2 = w_N^k.
DEVFN void sqpair(C* Vk, C* Vn, C t2) {
    C a = *Vk, b = conjg(*Vn);
    C E = chalf(cadd(a, b));
    C O = mulmi(chalf(csub(a, b)), 1);  // (a - b) / (2i)
    C P = cadd(cmul(E, E), cmul(t2, cmul(O, O)));
    C Q = cmul(E, O);
    Q = cadd(Q, Q);
    *Vk = cadd(P, muli(Q));
    *Vn = cadd(conjg(P), muli(conjg(Q)));
}
DEVFN void mulpair(C* Vk, C* Vn, C Uk, C Un, C t2) {
    C a = *Vk, b = conjg(*Vn);
    C Ex = chalf(cadd(a, b)), Ox = mulmi(chalf(csub(a, b)), 1);
    C c = Uk, d = conjg(Un);
    C Ey = chalf(cadd(c, d)), Oy = mulmi(chalf(csub(c, d)), 1);
    C E = cadd(cmul(Ex, Ey), cmul(t2, cmul(Ox, Oy)));
    C O = cadd(cmul(Ex, Oy), cmul(Ox, Ey));
    *Vk = cadd(E, muli(O));
    *Vn = cadd(conjg(E), muli(conjg(O)));
}

DEVFN uint tcol(uint g, uint arr) {
    uint c = g * TG + (arr < TG ? arr : arr - TG);
    return arr < TG ? c : (c == 0 ? W / 2 : W - c);
}
// mode 0: square, 1: forward column FFT only, 2: multiply by Bf (already forward)
DEVFN void tailImpl(LOCAL C* s1, GLOBAL C* A, GLOBAL const C* Bf, GLOBAL const C* twH, GLOBAL const C* twN1,
                    GLOBAL const C* twN2, const int mode) {
    const uint g = group_id(), t = local_id();
    for (uint idx = t; idx < 2 * TG * H; idx += WG) {
        uint arr = idx % (2 * TG), kh = idx / (2 * TG);
        s1[arr * H + kh] = A[kh * W + tcol(g, arr)];
    }
    BARRIER();
#define NB_ (2 * TG)
    FFTH_RUN(s1, twH, 1)
#undef NB_
    if (mode == 1) {
        for (uint idx = t; idx < 2 * TG * H; idx += WG) {
            uint arr = idx % (2 * TG), kh = idx / (2 * TG);
            A[kh * W + tcol(g, arr)] = s1[arr * H + kh];
        }
        return;
    }
    LOCAL C* r = s1;
    const uint ntask = TG * H + (g == 0 ? 1 : 0);
    for (uint u = t; u < ntask; u += WG) {
        uint ia, ib, colA, colB, kk;
        LOCAL C* pa;
        LOCAL C* pb;
        if (g == 0 && u <= H) {
            if (u <= H / 2) { ia = u; ib = (H - u) % H; pa = pb = r; colA = colB = 0; kk = W * u; }
            else { ia = u - (H / 2 + 1); ib = H - 1 - ia; pa = pb = r + TG * H; colA = colB = W / 2; kk = W / 2 + W * ia; }
        } else {
            uint uu = g == 0 ? u - 1 : u;
            uint i = uu / H, kh = uu % H;
            uint c = g * TG + i;
            ia = kh; ib = H - 1 - kh;
            pa = r + i * H; pb = r + (TG + i) * H;
            colA = c; colB = W - c; kk = c + W * kh;
        }
        C a = pa[ia], b = pb[ib];
        C tw2 = TWN(kk);
        if (mode == 0) sqpair(&a, &b, tw2);
        else mulpair(&a, &b, Bf[ia * W + colA], Bf[ib * W + colB], tw2);
        pa[ia] = a;
        if (!(pa == pb && ia == ib)) pb[ib] = b;
    }
    BARRIER();
#define NB_ (2 * TG)
    FFTH_RUN(s1, twH, -1)
#undef NB_
    for (uint idx = t; idx < 2 * TG * H; idx += WG) {
        uint arr = idx % (2 * TG), kh = idx / (2 * TG);
        A[kh * W + tcol(g, arr)] = s1[arr * H + kh];
    }
}

KERNEL tail(GLOBAL C* A, GLOBAL const C* twH, GLOBAL const C* twN1, GLOBAL const C* twN2) {
    LDS C s1[2 * TG * H];
    tailImpl(s1, A, A, twH, twN1, twN2, 0);
}
KERNEL tailFwd(GLOBAL C* A, GLOBAL const C* twH, GLOBAL const C* twN1, GLOBAL const C* twN2) {
    LDS C s1[2 * TG * H];
    tailImpl(s1, A, A, twH, twN1, twN2, 1);
}
KERNEL tailMul(GLOBAL C* A, GLOBAL const C* twH, GLOBAL const C* twN1, GLOBAL const C* twN2, GLOBAL const C* Bf) {
    LDS C s1[2 * TG * H];
    tailImpl(s1, A, Bf, twH, twN1, twN2, 2);
}

// exact value z from the FP32 approximation zf and the residue r = z mod p; err = distance of (zf - r)/p from an
// integer (must stay well below 0.5)
DEVFN long hround(float zf, uint r, float* err) {
    if (r == MP) r = 0;
    const float q = (zf - (float)r) * (1.0f / (float)MP);
    const float tq = RINTF(q);
    *err = FABSF(q - tq);
    return (long)r + (long)tq * (long)MP;
}

// inverse half for element m of value v: rounded digits (d0, d1) and the split carry; returns the round-off
DEVFN float unweigh(GLOBAL const df* WT, GLOBAL const uint* IT, uint m, C v, int mulc, int2* dig, long* cy) {
    uint ce0, ce1, ce2, ce4, rs0, rs1;
    uint B0 = Bpos2(IT, 2 * m, &ce0, &rs0), B1 = Bpos2(IT, 2 * m + 1, &ce1, &rs1), B2 = Bpos(IT, 2 * m + 2, &ce2);
    uint B4 = Bpos(IT, 2 * m + 4, &ce4);
    float e0, e1;
    long r0 = hround(v.x * iwgtF(WT, 2 * m, ce0), mmul(MRE(v), iwgtM(WT, 2 * m, ce0, rs0)), &e0);
    long r1 = hround(v.y * iwgtF(WT, 2 * m + 1, ce1), mmul(MIM(v), iwgtM(WT, 2 * m + 1, ce1, rs1)), &e1);
    long c0, c1;
    long d0 = bal(r0 * mulc, B1 - B0, &c0);
    long d1 = bal(r1 * mulc + c0, B2 - B1, &c1);
    *dig = I2((int)d0, (int)d1);
    // split the carry: low part (one element's worth) for m+1, the rest for m+2
    const uint e = B4 - B2;
    const long q2 = (c1 + (1L << (e - 1))) >> e;
    cy[0] = c1 - (q2 << e);
    cy[1] = q2;
    return FMAXF(e0, e1);
}

KERNEL invW(GLOBAL const C* A, GLOBAL DT* D, GLOBAL CS* CY, GLOBAL const df* IWT, GLOBAL const C* twW,
            GLOBAL const C* twN1, GLOBAL const C* twN2, const int mulc, GLOBAL uint* maxerr, GLOBAL const uint* IT) {
    LDS C s1[W];
    LDS uint lerr;
    const uint h = group_id(), t = local_id();
    if (t == 0) lerr = 0;
    for (uint kw = t; kw < W; kw += WG) s1[kw] = cmul(A[h * W + kw], conjg(TWN(h * kw)));
    BARRIER();
#define NB_ 1
    FFTW_RUN(s1, twW, -1)
#undef NB_
    float emax = 0;
    for (uint w = t; w < W; w += WG) {
        const uint pos = h * W + w;
        int2 dig;
        long cy[2];
        emax = FMAXF(emax, unweigh(IWT, IT, h + H * w, s1[w], mulc, &dig, cy));
        D[pos] = MKD(dig.x, dig.y);
        CY[pos] = packCY(cy[0], cy[1]);
    }
    atom_max(&lerr, (uint)(emax * 4294967295.0f));
    BARRIER();
    if (t == 0) atom_max(maxerr, lerr);
}
