// hyprp: weighted real FFT mod 2^n - c (four-step W x H, three kernels, same structure as dfprp), computed twice
// at once: once in FP32 (approximate) and once exactly in GF(p^2), p = M31 = 2^31 - 1 (a "complex" NTT; i^2 = -1).
// Every complex value is a float4: .xy = FP32 complex, .zw = the GF(p^2) element (bit pattern of two uints).
// The exact result z_j of the convolution is rebuilt as z = r + p * round((z_fp32 - r) / p) with r = z mod p
// from the NTT, so the FP32 part only has to be accurate to +-p/2 and words can be ~16-18 bits wide.
// GF(p) facts used: 2^31 == 1, so multiplying by a power of two is a 31-bit rotation; 1/2 = 2^30, 1/sqrt2 = 2^15;
// 2 and c = 25 have odd order, so the DWT weights 2^(k/L) and c^(j/L) exist in GF(p).
// Compile-time constants: W, H, NEXP, CC, TG, NH, WG, WSTAGES, HSTAGES, MU (= L^-1 mod 31), LOGN (= log2 N).
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

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
inline uint mred(uint x) { return (x & MP) + (x >> 31); }
inline uint madd(uint a, uint b) { return mred(a + b); }
inline uint msub(uint a, uint b) { return mred(a + (MP - b)); }
inline uint mneg(uint a) { return MP - a; }
inline uint mmul(uint a, uint b) {
    ulong t = (ulong)a * b;
    return mred((uint)(t & MP) + (uint)(t >> 31));
}
inline uint mrot(uint x, uint s) { return s ? (((x << s) & MP) | (x >> (31 - s))) : x; }  // x * 2^s, s < 31
inline uint mhalf(uint x) { return (x >> 1) | ((x & 1u) << 30); }                     // x / 2

// ---- hybrid complex values
inline uint2 MPART(C a) { return as_uint2(a.zw); }
inline C MKH(float2 f, uint2 m) { return (C)(f, as_float2(m)); }
inline C cadd(C a, C b) {
    uint2 x = MPART(a), y = MPART(b);
    return MKH(a.xy + b.xy, (uint2)(madd(x.x, y.x), madd(x.y, y.y)));
}
inline C csub(C a, C b) {
    uint2 x = MPART(a), y = MPART(b);
    return MKH(a.xy - b.xy, (uint2)(msub(x.x, y.x), msub(x.y, y.y)));
}
inline C cmul(C a, C b) {
    uint2 x = MPART(a), y = MPART(b);
    float2 f = (float2)(fma(a.x, b.x, -a.y * b.y), fma(a.x, b.y, a.y * b.x));
    uint2 m = (uint2)(msub(mmul(x.x, y.x), mmul(x.y, y.y)), madd(mmul(x.x, y.y), mmul(x.y, y.x)));
    return MKH(f, m);
}
inline C conjg(C a) { uint2 m = MPART(a); return MKH((float2)(a.x, -a.y), (uint2)(m.x, mneg(m.y))); }
// multiply by -i (forward) or +i (inverse)
inline C mulmi(C a, int dir) {
    uint2 m = MPART(a);
    return dir > 0 ? MKH((float2)(a.y, -a.x), (uint2)(m.y, mneg(m.x))) : MKH((float2)(-a.y, a.x), (uint2)(mneg(m.y), m.x));
}
inline C muli(C a) { uint2 m = MPART(a); return MKH((float2)(-a.y, a.x), (uint2)(mneg(m.y), m.x)); }
inline C chalf(C a) { uint2 m = MPART(a); return MKH(a.xy * 0.5f, (uint2)(mhalf(m.x), mhalf(m.y))); }
inline C csqrthalf(C a) { uint2 m = MPART(a); return MKH(a.xy * SQRT1_2F, (uint2)(mrot(m.x, 15), mrot(m.y, 15))); }
#define cadd_r cadd
#define csub_r csub
inline C cnorm(C a) { return a; }

inline void dft2(C* x) { C t = x[0]; x[0] = cadd(t, x[1]); x[1] = csub(t, x[1]); }
inline void dft4(C* x, int dir) {
    C v0 = cadd(x[0], x[2]), v1 = csub(x[0], x[2]), v2 = cadd(x[1], x[3]), v3 = mulmi(csub(x[1], x[3]), dir);
    x[0] = cadd(v0, v2); x[2] = csub(v0, v2); x[1] = cadd(v1, v3); x[3] = csub(v1, v3);
}
// the GF(p^2) twiddle tables are built so that w_P^(P/4) = -i and w_P^(P/8) = (1 - i)/sqrt2, as in C
inline void dft8(C* x, int dir) {
    C e[4] = {x[0], x[2], x[4], x[6]}, o[4] = {x[1], x[3], x[5], x[7]};
    dft4(e, dir); dft4(o, dir);
    C a = o[1], b = o[3];
    o[1] = csqrthalf(cadd(a, mulmi(a, dir)));
    o[3] = csqrthalf(csub(mulmi(b, dir), b));
    o[2] = mulmi(o[2], dir);
    for (int q = 0; q < 4; ++q) { x[q] = cadd(e[q], o[q]); x[q + 4] = csub(e[q], o[q]); }
}
// lengths with factors 3 or 5 are not supported by hyprp (no 5th roots of unity in GF(p^2)); stubs keep the
// shared stage macro compiling
inline void dft3(C* x, int dir) {}
inline void dft5(C* x, int dir) {}

// One Stockham stage, in place in local memory a (NB arrays of length P), radix R, sub-length PS.
// tw[j] = exp(-2 pi i j / P).
#define FFT_STAGE(a, P, NB, R, PS, tw, dir)                                                        \
    {                                                                                              \
        const uint nbf = (P) / (R), tot = (NB) * nbf;                                              \
        C x[((NB) * ((P) / (R)) + WG - 1) / WG][8];                                                               \
        _Pragma("unroll") for (uint u = 0; u < ((NB) * ((P) / (R)) + WG - 1) / WG; ++u) {                         \
            const uint g = t + u * WG;                                                             \
            if (g < tot) {                                                                         \
                const uint arr = g / nbf, i = g % nbf;                                             \
                _Pragma("unroll") for (uint r = 0; r < (R); ++r) x[u][r] = a[arr * (P) + i + r * nbf]; \
            }                                                                                      \
        }                                                                                          \
        barrier(CLK_LOCAL_MEM_FENCE);                                                              \
        _Pragma("unroll") for (uint u = 0; u < ((NB) * ((P) / (R)) + WG - 1) / WG; ++u) {                         \
            const uint g = t + u * WG;                                                             \
            if (g < tot) {                                                                         \
                const uint arr = g / nbf, i = g % nbf, k = i % (PS);                               \
                if ((PS) > 1) {                                                                    \
                    const uint stepi = k * ((P) / ((R) * (PS)));                                   \
                    _Pragma("unroll") for (uint r = 1; r < (R); ++r) {                             \
                        C w = tw[r * stepi];                                                       \
                        if ((dir) < 0) w = conjg(w);                                               \
                        x[u][r] = cmul(x[u][r], w);                                                \
                    }                                                                              \
                }                                                                                  \
                if ((R) == 8) dft8(x[u], dir);                                                     \
                else if ((R) == 4) dft4(x[u], dir);                                                \
                else if ((R) == 2) dft2(x[u]);                                                     \
                else if ((R) == 5) dft5(x[u], dir);                                                \
                else dft3(x[u], dir);                                                              \
                const uint j = (i - k) * (R) + k;                                                  \
                _Pragma("unroll") for (uint r = 0; r < (R); ++r) a[arr * (P) + j + r * (PS)] = x[u][r]; \
            }                                                                                      \
        }                                                                                          \
        barrier(CLK_LOCAL_MEM_FENCE);                                                              \
    }
// WSTAGES / HSTAGES expand to a list of STW(R, PS) / STH(R, PS) items (host generated). NB_ must be #defined
// to a literal around each use.
#define STW(R, PS) FFT_STAGE(a_, W, NB_, R, PS, tw_, dir_)
#define STH(R, PS) FFT_STAGE(a_, H, NB_, R, PS, tw_, dir_)
#define FFTW_RUN(a, tw, dir) { __local C* a_ = (a); __global const C* tw_ = (tw); const int dir_ = (dir); WSTAGES }
#define FFTH_RUN(a, tw, dir) { __local C* a_ = (a); __global const C* tw_ = (tw); const int dir_ = (dir); HSTAGES }

// ---- digit positions and weights.  B(j) = ceil(n j / L) = FH + FL + ceil((RH + RL) / L)
// itab = [RH(NH) | RL(WS) | FH(NH) | FL(WS)]
// wtab = [WH(NH) | WL(WS) | iWH(NH) | iWL(WS)], each entry (FP32 factor, bits of the GF(p) factor c^(.../L))
inline uint Bpos2(__global const uint* it, uint j, uint* ce, uint* rs) {
    const uint jh = j / WS, jl = j % WS;
    *rs = it[jh] + it[NH + jl];
    *ce = *rs == 0 ? 0u : (*rs <= LL ? 1u : 2u);
    return it[NH + WS + jh] + it[2 * NH + WS + jl] + *ce;
}
inline uint Bpos(__global const uint* it, uint j, uint* ce) { uint rs; return Bpos2(it, j, ce, &rs); }
// GF(p) weight 2^((ce L - rs)/L) = 2^(MU (ce L - rs) mod 31): a rotation
inline uint wrot(uint ce, uint rs) {
    long v = ((long)ce * LL - (long)rs) % 31;
    if (v < 0) v += 31;
    return (uint)((v * MU) % 31);
}
inline float wgtF(__global const df* wt, uint j, uint ce) { return ldexp(wt[j / WS].x * wt[NH + j % WS].x, (int)ce); }
inline float iwgtF(__global const df* wt, uint j, uint ce) {
    return ldexp(wt[NH + WS + j / WS].x * wt[2 * NH + WS + j % WS].x, -(int)ce);
}
inline uint wgtM(__global const df* wt, uint j, uint ce, uint rs) {
    return mrot(mmul(as_uint(wt[j / WS].y), as_uint(wt[NH + j % WS].y)), wrot(ce, rs));
}
inline uint iwgtM(__global const df* wt, uint j, uint ce, uint rs) {  // includes 1/N = 2^-LOGN
    return mrot(mmul(as_uint(wt[NH + WS + j / WS].y), as_uint(wt[2 * NH + WS + j % WS].y)),
                (62 - wrot(ce, rs) - (LOGN % 31)) % 31);
}

inline long bal(long v, uint b, long* carry) {
    const long base = 1L << b, hb = base >> 1;
    long d = ((v + hb) & (base - 1)) - hb;
    *carry = (v - d) >> b;
    return d;
}
inline uint tomod(long d) { return d < 0 ? (uint)(d + MP) : (uint)d; }  // |d| < p

// twN1[i] = w_N^i for i < W ; twN2[i] = w_N^(W i) = w_H^i for i < H
inline C twN(__global const C* twN1, __global const C* twN2, uint e) { return cmul(twN1[e % W], twN2[e / W]); }

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void fwdW(__global const int2* D, __global const int2* CY, __global const df* WT, __global C* A,
          __global const C* twW, __global const C* twN1, __global const C* twN2, __global const uint* IT) {
    __local C s1[W];
    const uint h = get_group_id(0), t = get_local_id(0);
    for (uint w = t; w < W; w += WG) {
        const uint pos = h * W + w;
        const uint m = h + H * w;
        // two-level carry: CY[e].x from element m-1, CY[e].y from element m-2 (both land at digit 2m)
        const uint m1 = m == 0 ? NC - 1 : m - 1;
        const uint m2 = m < 2 ? m + NC - 2 : m - 2;
        const int2 cA = CY[(m1 % H) * W + m1 / H], cB = CY[(m2 % H) * W + m2 / H];
        const long cin = (long)cA.x * (m == 0 ? CC : 1) + (long)cB.y * (m < 2 ? CC : 1);
        int2 d = D[pos];
        uint ce0, ce1, rs0, rs1;
        uint B0 = Bpos2(IT, 2 * m, &ce0, &rs0), B1 = Bpos2(IT, 2 * m + 1, &ce1, &rs1);
        long c0;
        long d0 = bal((long)d.x + cin, B1 - B0, &c0);
        long d1 = (long)d.y + c0;
        float2 f = (float2)((float)d0 * wgtF(WT, 2 * m, ce0), (float)d1 * wgtF(WT, 2 * m + 1, ce1));
        uint2 mm = (uint2)(mmul(tomod(d0), wgtM(WT, 2 * m, ce0, rs0)), mmul(tomod(d1), wgtM(WT, 2 * m + 1, ce1, rs1)));
        s1[w] = MKH(f, mm);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#define NB_ 1
    FFTW_RUN(s1, twW, 1)
#undef NB_
    for (uint kw = t; kw < W; kw += WG) A[h * W + kw] = cmul(s1[kw], twN(twN1, twN2, h * kw));
}

// square step for a pair (k, N-k): Vk, Vn in/out. t2 = w_N^k.
inline void sqpair(C* Vk, C* Vn, C t2) {
    C a = *Vk, b = conjg(*Vn);
    C E = chalf(cadd(a, b));
    C O = mulmi(chalf(csub(a, b)), 1);  // (a - b) / (2i)
    C P = cadd(cmul(E, E), cmul(t2, cmul(O, O)));
    C Q = cmul(E, O);
    Q = cadd(Q, Q);
    *Vk = cadd(P, muli(Q));
    *Vn = cadd(conjg(P), muli(conjg(Q)));
}
inline void mulpair(C* Vk, C* Vn, C Uk, C Un, C t2) {
    C a = *Vk, b = conjg(*Vn);
    C Ex = chalf(cadd(a, b)), Ox = mulmi(chalf(csub(a, b)), 1);
    C c = Uk, d = conjg(Un);
    C Ey = chalf(cadd(c, d)), Oy = mulmi(chalf(csub(c, d)), 1);
    C E = cadd(cmul(Ex, Ey), cmul(t2, cmul(Ox, Oy)));
    C O = cadd(cmul(Ex, Oy), cmul(Ox, Ey));
    *Vk = cadd(E, muli(O));
    *Vn = cadd(conjg(E), muli(conjg(O)));
}

inline uint tcol(uint g, uint arr) {
    uint c = g * TG + (arr < TG ? arr : arr - TG);
    return arr < TG ? c : (c == 0 ? W / 2 : W - c);
}
// mode 0: square, 1: forward column FFT only, 2: multiply by Bf (already forward)
inline __attribute__((always_inline)) void tailImpl(__local C* s1, __global C* A, __global const C* Bf,
                                                    __global const C* twH, __global const C* twN1,
                                                    __global const C* twN2, const int mode) {
    const uint g = get_group_id(0), t = get_local_id(0);
    for (uint idx = t; idx < 2 * TG * H; idx += WG) {
        uint arr = idx % (2 * TG), kh = idx / (2 * TG);
        s1[arr * H + kh] = A[kh * W + tcol(g, arr)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
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
    __local C* r = s1;
    const uint ntask = TG * H + (g == 0 ? 1 : 0);
    for (uint u = t; u < ntask; u += WG) {
        uint ia, ib, colA, colB, kk;
        __local C* pa;
        __local C* pb;
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
        C tw2 = twN(twN1, twN2, kk);
        if (mode == 0) sqpair(&a, &b, tw2);
        else mulpair(&a, &b, Bf[ia * W + colA], Bf[ib * W + colB], tw2);
        pa[ia] = a;
        if (!(pa == pb && ia == ib)) pb[ib] = b;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#define NB_ (2 * TG)
    FFTH_RUN(s1, twH, -1)
#undef NB_
    for (uint idx = t; idx < 2 * TG * H; idx += WG) {
        uint arr = idx % (2 * TG), kh = idx / (2 * TG);
        A[kh * W + tcol(g, arr)] = s1[arr * H + kh];
    }
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void tail(__global C* A, __global const C* twH, __global const C* twN1, __global const C* twN2) {
    __local C s1[2 * TG * H];
    tailImpl(s1, A, A, twH, twN1, twN2, 0);
}
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void tailFwd(__global C* A, __global const C* twH, __global const C* twN1, __global const C* twN2) {
    __local C s1[2 * TG * H];
    tailImpl(s1, A, A, twH, twN1, twN2, 1);
}
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void tailMul(__global C* A, __global const C* twH, __global const C* twN1, __global const C* twN2,
             __global const C* Bf) {
    __local C s1[2 * TG * H];
    tailImpl(s1, A, Bf, twH, twN1, twN2, 2);
}

// exact value z from the FP32 approximation zf and the residue r = z mod p; err = distance of (zf - r)/p from an
// integer (must stay well below 0.5)
inline long hround(float zf, uint r, float* err) {
    if (r == MP) r = 0;
    const float q = (zf - (float)r) * (1.0f / (float)MP);
    const float tq = rint(q);
    *err = fabs(q - tq);
    return (long)r + (long)tq * (long)MP;
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void invW(__global const C* A, __global int2* D, __global int2* CY, __global const df* IWT,
          __global const C* twW, __global const C* twN1, __global const C* twN2, const int mulc,
          __global uint* maxerr, __global const uint* IT) {
    __local C s1[W];
    __local uint lerr;
    const uint h = get_group_id(0), t = get_local_id(0);
    if (t == 0) lerr = 0;
    for (uint kw = t; kw < W; kw += WG) s1[kw] = cmul(A[h * W + kw], conjg(twN(twN1, twN2, h * kw)));
    barrier(CLK_LOCAL_MEM_FENCE);
#define NB_ 1
    FFTW_RUN(s1, twW, -1)
#undef NB_
    float emax = 0;
    for (uint w = t; w < W; w += WG) {
        const uint pos = h * W + w;
        const uint m = h + H * w;
        uint ce0, ce1, ce2, ce4, rs0, rs1;
        uint B0 = Bpos2(IT, 2 * m, &ce0, &rs0), B1 = Bpos2(IT, 2 * m + 1, &ce1, &rs1), B2 = Bpos(IT, 2 * m + 2, &ce2);
        uint B4 = Bpos(IT, 2 * m + 4, &ce4);
        C v = s1[w];
        uint2 vm = MPART(v);
        float e0, e1;
        long r0 = hround(v.x * iwgtF(IWT, 2 * m, ce0), mmul(vm.x, iwgtM(IWT, 2 * m, ce0, rs0)), &e0);
        long r1 = hround(v.y * iwgtF(IWT, 2 * m + 1, ce1), mmul(vm.y, iwgtM(IWT, 2 * m + 1, ce1, rs1)), &e1);
        emax = fmax(emax, fmax(e0, e1));
        long c0, c1;
        long d0 = bal(r0 * mulc, B1 - B0, &c0);
        long d1 = bal(r1 * mulc + c0, B2 - B1, &c1);
        D[pos] = (int2)((int)d0, (int)d1);
        // split the carry: low part (one element's worth) for m+1, the rest for m+2
        const uint e = B4 - B2;
        const long q2 = (c1 + (1L << (e - 1))) >> e;
        CY[pos] = (int2)((int)(c1 - (q2 << e)), (int)q2);
    }
    atomic_max(&lerr, (uint)(emax * 4294967295.0f));
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t == 0) atomic_max(maxerr, lerr);
}
