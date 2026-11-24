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

// Batched Stockham FFT in local memory. nb arrays of length P (log2 P = logP) stored contiguously
// in a; b is scratch of the same size. tw[j] = exp(-2 pi i j / P). Returns the buffer holding the result.
__local T2* fft_lds(__local T2* a, __local T2* b, const uint P, const uint logP, const uint nb,
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
inline T2 twN(__global const T2* twN1, __global const T2* twN2, uint e) {
    return cmul(twN1[e & (W - 1)], twN2[e / W]);
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void fwdW(__global const int2* D, __global const long* CY, __global const T2* WT, __global T2* A,
          __global const T2* twW, __global const T2* twN1, __global const T2* twN2) {
    __local T2 s1[W], s2[W];
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
        T2 wt = WT[pos];
        s1[w] = (T2)((double)d0 * wt.x, (double)d1 * wt.y);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __local T2* r = fft_lds(s1, s2, W, LOGW, 1, twW, 1);
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

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void tail(__global T2* A, __global const T2* twH, __global const T2* twN1, __global const T2* twN2) {
    __local T2 s1[2 * H], s2[2 * H];
    const uint g = get_group_id(0), t = get_local_id(0);
    const uint col1 = g, col2 = g == 0 ? W / 2 : W - g;
    for (uint i = t; i < 2 * H; i += WG) {
        uint col = i < H ? col1 : col2;
        s1[i] = A[(i & (H - 1)) * W + col];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __local T2* r = fft_lds(s1, s2, H, LOGH, 2, twH, 1);
    __local T2* x1 = r;
    __local T2* x2 = r + H;
    if (g != 0) {
        for (uint kh = t; kh < H; kh += WG) {
            T2 a = x1[kh], b = x2[H - 1 - kh];
            sqpair(&a, &b, twN(twN1, twN2, col1 + W * kh));
            x1[kh] = a;
            x2[H - 1 - kh] = b;
        }
    } else {
        // column 0: k = W kh pairs with (H - kh) mod H ; column W/2: kh pairs with H-1-kh
        for (uint u = t; u < H / 2 + 1 + H / 2; u += WG) {
            if (u <= H / 2) {
                uint kh = u, kn = (H - kh) & (H - 1);
                T2 a = x1[kh], b = x1[kn];
                sqpair(&a, &b, twN(twN1, twN2, W * kh));
                x1[kh] = a;
                if (kn != kh) x1[kn] = b;
            } else {
                uint kh = u - (H / 2 + 1);
                T2 a = x2[kh], b = x2[H - 1 - kh];
                sqpair(&a, &b, twN(twN1, twN2, W / 2 + W * kh));
                x2[kh] = a;
                x2[H - 1 - kh] = b;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __local T2* other = r == s1 ? s2 : s1;
    r = fft_lds(r, other, H, LOGH, 2, twH, -1);
    for (uint i = t; i < 2 * H; i += WG) {
        uint col = i < H ? col1 : col2;
        A[(i & (H - 1)) * W + col] = r[i];
    }
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void invW(__global const T2* A, __global int2* D, __global long* CY, __global const T2* IWT,
          __global const T2* twW, __global const T2* twN1, __global const T2* twN2, const int mulc,
          __global uint* maxerr) {
    __local T2 s1[W], s2[W];
    __local uint lerr;
    const uint h = get_group_id(0), t = get_local_id(0);
    if (t == 0) lerr = 0;
    for (uint kw = t; kw < W; kw += WG) s1[kw] = cmul(A[h * W + kw], conjg(twN(twN1, twN2, h * kw)));
    barrier(CLK_LOCAL_MEM_FENCE);
    __local T2* r = fft_lds(s1, s2, W, LOGW, 1, twW, -1);
    double emax = 0;
    for (uint w = t; w < W; w += WG) {
        const uint pos = h * W + w;
        const uint m = h + H * w;
        T2 iw = IWT[pos];
        double x0 = r[w].x * iw.x, x1 = r[w].y * iw.y;
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
