// dfprp: weighted real FFT mod 2^n - c (four-step W x H, three kernels), where every
// floating-point value is a "double-float": an unevaluated sum hi + lo of two FP32 numbers (~48-bit mantissa).
// Complex values are float4 (re.hi, re.lo, im.hi, im.lo). L = 2*W*H need not be a power of two (H may carry a
// factor 3 or 5). Compile-time constants: W, H, LOGW, NEXP, CC, TG, NH (weight table rows), WSTAGES, HSTAGES,
// and the double-float constants K_* (hex float pairs).
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL FP_CONTRACT OFF  // error-free transforms must not be fused

typedef float2 df;
typedef float4 C;
#define NC (W * H)
#define LL (2 * NC)
#define WG 64
#define WS 512u  // weight table split: j = jh * WS + jl

#define MKC(r, i) ((C)((r), (i)))

inline df fts(float a, float b) { float s = a + b; return (df)(s, b - (s - a)); }
inline df ts(float a, float b) {
    float s = a + b;
    float bb = s - a;
    return (df)(s, (a - (s - bb)) + (b - bb));
}
#ifdef ACCURATE_ADD
inline df dadd(df a, df b) {
    df s = ts(a.x, b.x), t = ts(a.y, b.y);
    s.y += t.x;
    s = fts(s.x, s.y);
    s.y += t.y;
    return fts(s.x, s.y);
}
inline df dsub(df a, df b) { return dadd(a, (df)(-b.x, -b.y)); }
#else
inline df dadd(df a, df b) { df s = ts(a.x, b.x); return fts(s.x, s.y + (a.y + b.y)); }
inline df dsub(df a, df b) { df s = ts(a.x, -b.x); return fts(s.x, s.y + (a.y - b.y)); }
#endif
inline df dneg(df a) { return (df)(-a.x, -a.y); }
inline df dmul(df a, df b) {
    float p = a.x * b.x;
    float e = fma(a.x, b.x, -p);
    e = fma(a.x, b.y, fma(a.y, b.x, e));
    return fts(p, e);
}
inline df dmulf(df a, float b) {
    float p = a.x * b;
    float e = fma(a.x, b, -p);
    e = fma(a.y, b, e);
    return fts(p, e);
}
// a*b - c*d with a single renormalisation
inline df dmms(df a, df b, df c, df d) {
    float p1 = a.x * b.x, e1 = fma(a.x, b.x, -p1);
    float p2 = c.x * d.x, e2 = fma(c.x, d.x, -p2);
    df s = ts(p1, -p2);
    float lo = s.y + (e1 - e2);
    lo = fma(a.x, b.y, lo);
    lo = fma(a.y, b.x, lo);
    lo = fma(-c.x, d.y, lo);
    lo = fma(-c.y, d.x, lo);
    return fts(s.x, lo);
}
inline df dmma(df a, df b, df c, df d) { return dmms(a, b, dneg(c), d); }

inline C cadd(C a, C b) { return MKC(dadd(a.xy, b.xy), dadd(a.zw, b.zw)); }
inline C csub(C a, C b) { return MKC(dsub(a.xy, b.xy), dsub(a.zw, b.zw)); }
inline C cmul(C a, C b) { return MKC(dmms(a.xy, b.xy, a.zw, b.zw), dmma(a.xy, b.zw, a.zw, b.xy)); }
inline C csq(C a) { df t = dmul(a.xy, a.zw); return MKC(dmms(a.xy, a.xy, a.zw, a.zw), t * 2.0f); }
inline C cmulr(C a, df s) { return MKC(dmul(a.xy, s), dmul(a.zw, s)); }
inline C conjg(C a) { return (C)(a.x, a.y, -a.z, -a.w); }
// multiply by -i (forward) or +i (inverse)
inline C mulmi(C a, int dir) { return dir > 0 ? (C)(a.z, a.w, -a.x, -a.y) : (C)(-a.z, -a.w, a.x, a.y); }
inline C muli(C a) { return (C)(-a.z, -a.w, a.x, a.y); }

#define KDF(name) ((df)(name##_HI, name##_LO))

inline void dft2(C* x) { C t = x[0]; x[0] = cadd(t, x[1]); x[1] = csub(t, x[1]); }
inline void dft4(C* x, int dir) {
    C v0 = cadd(x[0], x[2]), v1 = csub(x[0], x[2]), v2 = cadd(x[1], x[3]), v3 = mulmi(csub(x[1], x[3]), dir);
    x[0] = cadd(v0, v2); x[2] = csub(v0, v2); x[1] = cadd(v1, v3); x[3] = csub(v1, v3);
}
inline void dft8(C* x, int dir) {
    C e[4] = {x[0], x[2], x[4], x[6]}, o[4] = {x[1], x[3], x[5], x[7]};
    dft4(e, dir); dft4(o, dir);
    const df h = KDF(K_SQRT1_2);
    // o1 *= (1 -/+ i)/sqrt2, o3 *= (-1 -/+ i)/sqrt2, o2 *= -/+ i
    C a = o[1], b = o[3];
    C ma = mulmi(a, dir), mb = mulmi(b, dir);
    o[1] = cmulr(cadd(a, ma), h);
    o[3] = cmulr(csub(mb, b), h);
    o[2] = mulmi(o[2], dir);
    for (int q = 0; q < 4; ++q) { x[q] = cadd(e[q], o[q]); x[q + 4] = csub(e[q], o[q]); }
}
inline void dft3(C* x, int dir) {
    C s = cadd(x[1], x[2]);
    C d = mulmi(cmulr(csub(x[1], x[2]), KDF(K_S3)), dir);  // -/+ i * (sqrt3/2)(x1 - x2)
    C t = csub(x[0], s * 0.5f);
    x[0] = cadd(x[0], s);
    x[1] = cadd(t, d);
    x[2] = csub(t, d);
}
inline void dft5(C* x, int dir) {
    C t1 = cadd(x[1], x[4]), t2 = cadd(x[2], x[3]), t3 = csub(x[1], x[4]), t4 = csub(x[2], x[3]);
    const df c1 = KDF(K_C1), c2 = KDF(K_C2), s1 = KDF(K_S1), s2 = KDF(K_S2);
    C a1 = cadd(x[0], cadd(cmulr(t1, c1), cmulr(t2, c2)));
    C a2 = cadd(x[0], cadd(cmulr(t1, c2), cmulr(t2, c1)));
    C b1 = mulmi(cadd(cmulr(t3, s1), cmulr(t4, s2)), dir);
    C b2 = mulmi(csub(cmulr(t3, s2), cmulr(t4, s1)), dir);
    x[0] = cadd(x[0], cadd(t1, t2));
    x[1] = cadd(a1, b1);
    x[4] = csub(a1, b1);
    x[2] = cadd(a2, b2);
    x[3] = csub(a2, b2);
}

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
// itab = [RH(NH) | RL(WS) | FH(NH) | FL(WS)], wtab = [WH(NH) | WL(WS) | iWH(NH) | iWL(WS)] (df)
inline uint Bpos(__global const uint* it, uint j, uint* ce) {
    const uint jh = j / WS, jl = j % WS;
    const uint rs = it[jh] + it[NH + jl];
    *ce = rs == 0 ? 0u : (rs <= LL ? 1u : 2u);
    return it[NH + WS + jh] + it[2 * NH + WS + jl] + *ce;
}
inline df dldexp(df a, int e) { return (df)(ldexp(a.x, e), ldexp(a.y, e)); }
inline df wgt(__global const df* wt, uint j, uint ce) { return dldexp(dmul(wt[j / WS], wt[NH + j % WS]), (int)ce); }
inline df iwgt(__global const df* wt, uint j, uint ce) {
    return dldexp(dmul(wt[NH + WS + j / WS], wt[2 * NH + WS + j % WS]), -(int)ce);
}

inline long bal(long v, uint b, long* carry) {
    const long base = 1L << b, hb = base >> 1;
    long d = ((v + hb) & (base - 1)) - hb;
    *carry = (v - d) >> b;
    return d;
}

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
        uint ce0, ce1;
        uint B0 = Bpos(IT, 2 * m, &ce0), B1 = Bpos(IT, 2 * m + 1, &ce1);
        long c0;
        long d0 = bal((long)d.x + cin, B1 - B0, &c0);
        long d1 = (long)d.y + c0;
        s1[w] = MKC(dmulf(wgt(WT, 2 * m, ce0), (float)d0), dmulf(wgt(WT, 2 * m + 1, ce1), (float)d1));
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
    C E = cadd(a, b) * 0.5f;
    C dd = csub(a, b) * 0.5f;
    C O = (C)(dd.z, dd.w, -dd.x, -dd.y);
    C P = cadd(csq(E), cmul(t2, csq(O)));
    C Q = cmul(E, O) * 2.0f;
    *Vk = cadd(P, muli(Q));
    *Vn = cadd(conjg(P), muli(conjg(Q)));
}
inline void mulpair(C* Vk, C* Vn, C Uk, C Un, C t2) {
    C a = *Vk, b = conjg(*Vn);
    C Ex = cadd(a, b) * 0.5f, dx = csub(a, b) * 0.5f;
    C Ox = (C)(dx.z, dx.w, -dx.x, -dx.y);
    C c = Uk, d = conjg(Un);
    C Ey = cadd(c, d) * 0.5f, dy = csub(c, d) * 0.5f;
    C Oy = (C)(dy.z, dy.w, -dy.x, -dy.y);
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

// round a double-float to the nearest integer; returns |fractional distance|
inline long dround(df x, float* err) {
    float rh = rint(x.x);
    float f = (x.x - rh) + x.y;
    float rf = rint(f);
    *err = fabs(f - rf);
    return (long)rh + (long)rf;
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
        uint ce0, ce1, ce2, ce4;
        uint B0 = Bpos(IT, 2 * m, &ce0), B1 = Bpos(IT, 2 * m + 1, &ce1), B2 = Bpos(IT, 2 * m + 2, &ce2);
        uint B4 = Bpos(IT, 2 * m + 4, &ce4);
        C v = s1[w];
        float e0, e1;
        long r0 = dround(dmul(v.xy, iwgt(IWT, 2 * m, ce0)), &e0);
        long r1 = dround(dmul(v.zw, iwgt(IWT, 2 * m + 1, ce1)), &e1);
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
