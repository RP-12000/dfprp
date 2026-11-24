// GPU Fermat PRP for N = 2^n - c (c small, positive), OpenCL FP64.
// Weighted (IBDWT-style) real FFT: digit j has weight 2^(ceil(nj/L)-nj/L) * c^(j/L), so a plain
// cyclic convolution of length L gives the product modulo 2^n - c directly (no zero padding).
// L real words are packed as N = L/2 complex values, N = W*H, four-step FFT:
//   fwdW : carry-in + weight + FFT over w (line) + twiddle      (one work-group per line h)
//   tail : FFT over h (column) + real-FFT square + inverse FFT  (one work-group per column pair)
//   invW : twiddle + inverse FFT over w + unweight + round + carry
// Complex element m = h + H*w holds digits 2m (re) and 2m+1 (im); it lives at memory pos h*W + w.
// Compile-time constants: W, H, LOGL, NEXP (n), CC (c).
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

typedef double2 T2;
#define NC (W * H)
#define LL (2 * NC)
#define WG 64
#ifdef HACK_CONTIG
#define TIDX(row, col) ((col) * H + (row))
#else
#define TIDX(row, col) ((row) * W + (col))
#endif
#ifdef HACK_NOFFT
#define FFT_LDS(a, b, P, lP, nb, tw, d) (a)
#elif defined(OLDFFT)
#define FFT_LDS fft_lds
#else
#define FFT_LDS(a, b, P, lP, nb, tw, d) fft_ip(a, P, lP, nb, tw, d)
#endif
#ifdef OLDFFT
#define LDS2(name, sz) __local T2 name[sz]
#else
#define LDS2(name, sz) __local T2* name = 0
#endif

inline T2 cmul(T2 a, T2 b) { return (T2)(fma(a.x, b.x, -a.y * b.y), fma(a.x, b.y, a.y * b.x)); }
inline T2 conjg(T2 a) { return (T2)(a.x, -a.y); }
inline T2 sq(T2 a) { return (T2)(fma(a.x, a.x, -a.y * a.y), 2 * a.x * a.y); }
// multiply by -i (forward) or +i (inverse)
inline T2 mulmi(T2 a, int dir) { return dir > 0 ? (T2)(a.y, -a.x) : (T2)(-a.y, a.x); }

// bits of digit j: ceil(n(j+1)/L) - ceil(nj/L)
inline uint dbits(uint j) {
    ulong a = (ulong)NEXP * j;
    ulong b = a + NEXP;
    return (uint)(((b + LL - 1) >> LOGL) - ((a + LL - 1) >> LOGL));
}

inline long bal(long v, uint b, long* carry) {
    const long base = 1L << b, hb = base >> 1;
    long d = ((v + hb) & (base - 1)) - hb;
    *carry = (v - d) >> b;
    return d;
}

#define SQRT1_2 0.70710678118654752440

inline void dft2(T2* x) { T2 t = x[0]; x[0] = t + x[1]; x[1] = t - x[1]; }
inline void dft4(T2* x, int dir) {
    T2 v0 = x[0] + x[2], v1 = x[0] - x[2], v2 = x[1] + x[3], v3 = mulmi(x[1] - x[3], dir);
    x[0] = v0 + v2; x[2] = v0 - v2; x[1] = v1 + v3; x[3] = v1 - v3;
}
inline void dft8(T2* x, int dir) {
    T2 e[4] = {x[0], x[2], x[4], x[6]}, o[4] = {x[1], x[3], x[5], x[7]};
    dft4(e, dir); dft4(o, dir);
    // o[q] *= w8^q
    T2 a = o[1], b = o[3];
    if (dir > 0) {
        o[1] = (T2)((a.x + a.y) * SQRT1_2, (a.y - a.x) * SQRT1_2);   // * (1-i)/sqrt2
        o[3] = (T2)((b.y - b.x) * SQRT1_2, -(b.x + b.y) * SQRT1_2);  // * (-1-i)/sqrt2
    } else {
        o[1] = (T2)((a.x - a.y) * SQRT1_2, (a.y + a.x) * SQRT1_2);   // * (1+i)/sqrt2
        o[3] = (T2)(-(b.x + b.y) * SQRT1_2, (b.x - b.y) * SQRT1_2);  // * (-1+i)/sqrt2
    }
    o[2] = mulmi(o[2], dir);
    for (int q = 0; q < 4; ++q) { x[q] = e[q] + o[q]; x[q + 4] = e[q] - o[q]; }
}

// In-place variant: one LDS buffer, each thread keeps its butterflies in registers between a read
// phase and a write phase (radix fixed per branch so register arrays are statically indexed).
#define FFT_STAGE(LR, PER)                                                                     \
    {                                                                                          \
        const uint R = 1u << LR, nbf = P >> LR, tot = nb * nbf;                                \
        T2 x[PER][1 << LR];                                                                    \
        _Pragma("unroll") for (uint u = 0; u < PER; ++u) {                                     \
            const uint g = t + u * WG;                                                         \
            if (g < tot) {                                                                     \
                const uint arr = g >> (logP - LR), i = g & (nbf - 1);                          \
                _Pragma("unroll") for (uint r = 0; r < R; ++r) x[u][r] = a[arr * P + i + r * nbf]; \
            }                                                                                  \
        }                                                                                      \
        barrier(CLK_LOCAL_MEM_FENCE);                                                          \
        _Pragma("unroll") for (uint u = 0; u < PER; ++u) {                                     \
            const uint g = t + u * WG;                                                         \
            if (g < tot) {                                                                     \
                const uint arr = g >> (logP - LR), i = g & (nbf - 1), k = i & (p - 1);         \
                if (p > 1) {                                                                   \
                    const uint stepi = k << (logP - LR - logp);                                \
                    _Pragma("unroll") for (uint r = 1; r < R; ++r) {                           \
                        T2 w = tw[(r * stepi) & (P - 1)];                                      \
                        if (dir < 0) w.y = -w.y;                                               \
                        x[u][r] = cmul(x[u][r], w);                                            \
                    }                                                                          \
                }                                                                              \
                if (LR == 3) dft8(x[u], dir);                                                  \
                else if (LR == 2) dft4(x[u], dir);                                             \
                else dft2(x[u]);                                                               \
                const uint j = ((i - k) << LR) + k;                                            \
                _Pragma("unroll") for (uint r = 0; r < R; ++r) a[arr * P + j + r * p] = x[u][r]; \
            }                                                                                  \
        }                                                                                      \
        barrier(CLK_LOCAL_MEM_FENCE);                                                          \
        p <<= LR;                                                                              \
        logp += LR;                                                                            \
    }

// requires nb * P <= 16 * WG
inline __attribute__((always_inline)) __local T2* fft_ip(__local T2* a, const uint P, const uint logP, const uint nb, __global const T2* tw,
                   const int dir) {
    const uint t = get_local_id(0);
    uint p = 1, logp = 0;
    while (logP - logp >= 3) FFT_STAGE(3, 2)
    if (logP - logp == 2) FFT_STAGE(2, 4)
    if (logP - logp == 1) FFT_STAGE(1, 8)
    return a;
}

// Batched Stockham FFT in local memory. nb arrays of length P (log2 P = logP) stored contiguously
// in a; b is scratch of the same size. tw[j] = exp(-2 pi i j / P). Returns the buffer holding the result.
inline __attribute__((always_inline)) __local T2* fft_lds(__local T2* a, __local T2* b, const uint P, const uint logP, const uint nb,
                    __global const T2* tw, const int dir) {
    const uint t = get_local_id(0);
    uint p = 1, logp = 0;
    while (p < P) {
        const uint left = logP - logp;
        const uint lr = left >= 3 ? 3 : left;  // radix 8, then 4 or 2
        const uint R = 1u << lr;
        const uint nbf = P >> lr;  // butterflies per array
        for (uint g = t; g < nb * nbf; g += WG) {
            const uint arr = g >> (logP - lr);
            const uint i = g & (nbf - 1);
            const uint k = i & (p - 1);
            __local const T2* in = a + arr * P;
            T2 x[8];
            for (uint r = 0; r < R; ++r) x[r] = in[i + r * nbf];
            if (p > 1) {
                // twiddle w_{Rp}^{rk} = w_P^{r k P/(R p)}
                const uint stepi = k << (logP - lr - logp);
                for (uint r = 1; r < R; ++r) {
                    T2 w = tw[(r * stepi) & (P - 1)];
                    if (dir < 0) w.y = -w.y;
                    x[r] = cmul(x[r], w);
                }
            }
            if (lr == 3) dft8(x, dir);
            else if (lr == 2) dft4(x, dir);
            else dft2(x);
            __local T2* out = b + arr * P;
            const uint j = ((i - k) << lr) + k;
            for (uint r = 0; r < R; ++r) out[j + r * p] = x[r];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        __local T2* tmp = a; a = b; b = tmp;
        p <<= lr;
        logp += lr;
    }
    return a;
}

// twN1[i] = w_N^i for i < W ; twN2[i] = w_N^(W i) = w_H^i for i < H
// weight of digit j from 4 small tables: 2^(r/L) * c^(j/L), r = (-n j) mod L. inv: reciprocal tables (incl. 1/N)
inline double wgt(__global const double* tb, uint j, uint inv) {
    const uint r = (0u - (uint)NEXP * j) & (LL - 1);
    __global const double* t = tb + (inv << (WSPLIT + 2));
    return (t[r & ((1u << WSPLIT) - 1)] * t[(1u << WSPLIT) + (r >> WSPLIT)]) *
           (t[(2u << WSPLIT) + (j & ((1u << WSPLIT) - 1))] * t[(3u << WSPLIT) + (j >> WSPLIT)]);
}

inline T2 twN(__global const T2* twN1, __global const T2* twN2, uint e) {
    return cmul(twN1[e & (W - 1)], twN2[e / W]);
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void fwdW(__global const int2* D, __global const long* CY, __global const double* WT, __global T2* A,
          __global const T2* twW, __global const T2* twN1, __global const T2* twN2) {
    __local T2 s1[W];
    LDS2(s2, W);
    const uint h = get_group_id(0), t = get_local_id(0);
    for (uint w = t; w < W; w += WG) {
        const uint pos = h * W + w;
        const uint m = h + H * w;
        long cin;
        if (h > 0) cin = CY[pos - W];
        else if (w > 0) cin = CY[(H - 1) * W + w - 1];
        else cin = CY[(H - 1) * W + W - 1] * CC;
        int2 d = D[pos];
        long c0;
        long d0 = bal((long)d.x + cin, dbits(2 * m), &c0);
        long d1 = (long)d.y + c0;
        s1[w] = (T2)((double)d0 * wgt(WT, 2 * m, 0), (double)d1 * wgt(WT, 2 * m + 1, 0));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __local T2* r = FFT_LDS(s1, s2, W, LOGW, 1, twW, 1);
    for (uint kw = t; kw < W; kw += WG) A[h * W + kw] = cmul(r[kw], twN(twN1, twN2, h * kw));
}

// square step for a pair (k, N-k): Vk, Vn in/out. t2 = w_N^k.
inline void sqpair(T2* Vk, T2* Vn, T2 t2) {
    T2 a = *Vk, b = conjg(*Vn);
    T2 E = (a + b) * 0.5;
    T2 dd = a - b;
    T2 O = (T2)(dd.y * 0.5, -dd.x * 0.5);
    T2 P = sq(E) + cmul(t2, sq(O));
    T2 Q = cmul(E, O);
    *Vk = P + (T2)(-2 * Q.y, 2 * Q.x);
    *Vn = conjg(P) + (T2)(2 * Q.y, 2 * Q.x);  // 2i * conj(Q)
}

// multiply step: (Vk, Vn) *= (Uk, Un) as real-FFT spectra
inline void mulpair(T2* Vk, T2* Vn, T2 Uk, T2 Un, T2 t2) {
    T2 a = *Vk, b = conjg(*Vn);
    T2 Ex = (a + b) * 0.5, dx = a - b;
    T2 Ox = (T2)(dx.y * 0.5, -dx.x * 0.5);
    T2 c = Uk, d = conjg(Un);
    T2 Ey = (c + d) * 0.5, dy = c - d;
    T2 Oy = (T2)(dy.y * 0.5, -dy.x * 0.5);
    T2 E = cmul(Ex, Ey) + cmul(t2, cmul(Ox, Oy));
    T2 O = cmul(Ex, Oy) + cmul(Ox, Ey);
    *Vk = E + (T2)(-O.y, O.x);
    *Vn = conjg(E) + (T2)(O.y, O.x);  // conj(E) + i conj(O)
}

// mode 0: square, mode 1: forward column FFT only (A -> A), mode 2: multiply by Bf (already forward)
// A work-group handles TG adjacent columns c = g*TG + i and their partners W - c (column 0's partner
// slot holds the self-paired column W/2). LDS layout: [2*TG arrays][H].
inline uint tcol(uint g, uint arr) {
    uint c = g * TG + (arr < TG ? arr : arr - TG);
    return arr < TG ? c : (c == 0 ? W / 2 : W - c);
}
inline __attribute__((always_inline)) void tailImpl(__local T2* s1, __local T2* s2, __global T2* A, __global const T2* Bf, __global const T2* twH,
                     __global const T2* twN1, __global const T2* twN2, const int mode) {
    const uint g = get_group_id(0), t = get_local_id(0);
    for (uint idx = t; idx < 2 * TG * H; idx += WG) {
        uint arr = idx % (2 * TG), kh = idx / (2 * TG);
        s1[arr * H + kh] = A[TIDX(kh, tcol(g, arr))];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __local T2* r = FFT_LDS(s1, s2, H, LOGH, 2 * TG, twH, 1);
    if (mode == 1) {
        for (uint idx = t; idx < 2 * TG * H; idx += WG) {
            uint arr = idx % (2 * TG), kh = idx / (2 * TG);
            A[TIDX(kh, tcol(g, arr))] = r[arr * H + kh];
        }
        return;
    }
    const uint ntask = TG * H + (g == 0 ? 1 : 0);
    for (uint u = t; u < ntask; u += WG) {
        uint ia, ib, colA, colB, kk;
        __local T2* pa;
        __local T2* pb;
        if (g == 0 && u <= H) {
            // column 0 self pairs (kh, H-kh), then column W/2 (partner slot 0) pairs (kh, H-1-kh)
            if (u <= H / 2) { ia = u; ib = (H - u) & (H - 1); pa = pb = r; colA = colB = 0; kk = W * u; }
            else { ia = u - (H / 2 + 1); ib = H - 1 - ia; pa = pb = r + TG * H; colA = colB = W / 2; kk = W / 2 + W * ia; }
        } else {
            uint uu = g == 0 ? u - 1 : u;  // skip the extra task of group 0
            uint i = uu / H, kh = uu % H;
            uint c = g * TG + i;
            ia = kh; ib = H - 1 - kh;
            pa = r + i * H; pb = r + (TG + i) * H;
            colA = c; colB = W - c; kk = c + W * kh;
        }
        T2 a = pa[ia], b = pb[ib];
        T2 tw2 = twN(twN1, twN2, kk);
        if (mode == 0) sqpair(&a, &b, tw2);
        else mulpair(&a, &b, Bf[TIDX(ia, colA)], Bf[TIDX(ib, colB)], tw2);
        pa[ia] = a;
        if (!(pa == pb && ia == ib)) pb[ib] = b;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    r = FFT_LDS(r, s2, H, LOGH, 2 * TG, twH, -1);
    for (uint idx = t; idx < 2 * TG * H; idx += WG) {
        uint arr = idx % (2 * TG), kh = idx / (2 * TG);
        A[TIDX(kh, tcol(g, arr))] = r[arr * H + kh];
    }
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void tail(__global T2* A, __global const T2* twH, __global const T2* twN1, __global const T2* twN2) {
    __local T2 s1[2 * TG * H];
    LDS2(s2, 2 * TG * H);
    tailImpl(s1, s2, A, A, twH, twN1, twN2, 0);
}
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void tailFwd(__global T2* A, __global const T2* twH, __global const T2* twN1, __global const T2* twN2) {
    __local T2 s1[2 * TG * H];
    LDS2(s2, 2 * TG * H);
    tailImpl(s1, s2, A, A, twH, twN1, twN2, 1);
}
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void tailMul(__global T2* A, __global const T2* twH, __global const T2* twN1, __global const T2* twN2,
             __global const T2* Bf) {
    __local T2 s1[2 * TG * H];
    LDS2(s2, 2 * TG * H);
    tailImpl(s1, s2, A, Bf, twH, twN1, twN2, 2);
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void invW(__global const T2* A, __global int2* D, __global long* CY, __global const double* IWT,
          __global const T2* twW, __global const T2* twN1, __global const T2* twN2, const int mulc,
          __global uint* maxerr) {
    __local T2 s1[W];
    LDS2(s2, W);
    __local uint lerr;
    const uint h = get_group_id(0), t = get_local_id(0);
    if (t == 0) lerr = 0;
    for (uint kw = t; kw < W; kw += WG) s1[kw] = cmul(A[h * W + kw], conjg(twN(twN1, twN2, h * kw)));
    barrier(CLK_LOCAL_MEM_FENCE);
    __local T2* r = FFT_LDS(s1, s2, W, LOGW, 1, twW, -1);
    double emax = 0;
    for (uint w = t; w < W; w += WG) {
        const uint pos = h * W + w;
        const uint m = h + H * w;
        double x0 = r[w].x * wgt(IWT, 2 * m, 1), x1 = r[w].y * wgt(IWT, 2 * m + 1, 1);
        double r0 = rint(x0), r1 = rint(x1);
        emax = fmax(emax, fmax(fabs(x0 - r0), fabs(x1 - r1)));
        long c0, c1;
        long d0 = bal((long)r0 * mulc, dbits(2 * m), &c0);
        long d1 = bal((long)r1 * mulc + c0, dbits(2 * m + 1), &c1);
        D[pos] = (int2)((int)d0, (int)d1);
        CY[pos] = c1;
    }
    atomic_max(&lerr, (uint)(emax * 4294967295.0));
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t == 0) atomic_max(maxerr, lerr);
}
