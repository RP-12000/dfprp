// prp4: Fermat PRP for N = 2^n - c with an integer NTT over three 31-bit primes (CRT), 32-bit limbs,
// zero-padded (acyclic) squaring, then folding with 2^(32*WN) == K = c * 2^(32*WN - n) (mod N).
// Length L = W*H (powers of two). Four-step layout like prp2: element m = h + H*w lives at pos h*W + w.
//   fwdW : load limbs (+ lazy carries, + K*T fold of the top) -> NTT over w (3 primes) -> twiddle
//   tail : NTT over h, square (or multiply), inverse NTT over h       (TG columns per work-group)
//   invW : inverse twiddle, inverse NTT over w, scale, CRT -> 96-bit coefficient z_m (times mulc)
//   carry: y_j = z_j + K z_(j+WN); limb sums; carry chains over SEG rows of one column (j consecutive)
// Host-provided defines: W, H, WN, KLO, KHI, NCOL, SEG, TG, prime constants P0..P2, PINV0..2, S0..2 (scale),
// CRT constants, NEXP, CC.
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#define LL (W * H)
#define WG 64
#define NSEG (H / SEG)

inline uint mm(uint a, uint b, uint p, uint pinv) {  // a*b/2^32 mod p (Montgomery)
    ulong t = (ulong)a * b;
    uint m = (uint)t * pinv;
    uint u = (uint)((t + (ulong)m * p) >> 32);
    return u >= p ? u - p : u;
}
inline uint ad(uint a, uint b, uint p) { uint s = a + b; return s >= p ? s - p : s; }
inline uint sb(uint a, uint b, uint p) { return a >= b ? a - b : a + p - b; }

#define PR(i) ((i) == 0 ? P0 : (i) == 1 ? P1 : P2)
#define PI(i) ((i) == 0 ? PINV0 : (i) == 1 ? PINV1 : PINV2)

// tables (uint, Montgomery form), per prime i (stride per prime given below):
//   twW[i][dir][W], twH[i][dir][H], twN1[i][dir][W] = w_L^(+-j), twN2[i][dir][H] = w_L^(+-W j)
// dir index 0 = forward, 1 = inverse.

// radix-4 / radix-2 Stockham stage on NB arrays of length P in local memory; array a belongs to prime a / (NB/3)
#define NTT_STAGE(a, P, NB, LR, PS, TW, dirI)                                                     \
    {                                                                                             \
        const uint R = 1u << (LR), nbf = (P) >> (LR), tot = (NB) * nbf;                           \
        uint x[((NB) * ((P) >> (LR)) + WG - 1) / WG][4];                                          \
        _Pragma("unroll") for (uint u = 0; u < ((NB) * ((P) >> (LR)) + WG - 1) / WG; ++u) {       \
            const uint g = t + u * WG;                                                            \
            if (g < tot) {                                                                        \
                const uint arr = g / nbf, i = g % nbf;                                            \
                _Pragma("unroll") for (uint r = 0; r < R; ++r) x[u][r] = a[arr * (P) + i + r * nbf]; \
            }                                                                                     \
        }                                                                                         \
        barrier(CLK_LOCAL_MEM_FENCE);                                                             \
        _Pragma("unroll") for (uint u = 0; u < ((NB) * ((P) >> (LR)) + WG - 1) / WG; ++u) {       \
            const uint g = t + u * WG;                                                            \
            if (g < tot) {                                                                        \
                const uint arr = g / nbf, i = g % nbf, k = i % (PS);                              \
                const uint pi = arr / ((NB) / 3);                                                 \
                const uint p = PR(pi), pinv = PI(pi);                                             \
                __global const uint* tw = TW(pi, dirI);                                           \
                if ((PS) > 1) {                                                                   \
                    const uint stepi = k * ((P) / (R * (PS)));                                    \
                    _Pragma("unroll") for (uint r = 1; r < R; ++r) x[u][r] = mm(x[u][r], tw[r * stepi], p, pinv); \
                }                                                                                 \
                if ((LR) == 2) {                                                                  \
                    const uint I = tw[(P) / 4];                                                   \
                    uint v0 = ad(x[u][0], x[u][2], p), v1 = sb(x[u][0], x[u][2], p);              \
                    uint v2 = ad(x[u][1], x[u][3], p), v3 = mm(sb(x[u][1], x[u][3], p), I, p, pinv); \
                    x[u][0] = ad(v0, v2, p); x[u][2] = sb(v0, v2, p);                             \
                    x[u][1] = ad(v1, v3, p); x[u][3] = sb(v1, v3, p);                             \
                } else {                                                                          \
                    uint v0 = x[u][0];                                                            \
                    x[u][0] = ad(v0, x[u][1], p); x[u][1] = sb(v0, x[u][1], p);                   \
                }                                                                                 \
                const uint j = ((i - k) << (LR)) + k;                                             \
                _Pragma("unroll") for (uint r = 0; r < R; ++r) a[arr * (P) + j + r * (PS)] = x[u][r]; \
            }                                                                                     \
        }                                                                                         \
        barrier(CLK_LOCAL_MEM_FENCE);                                                             \
    }

// WSTAGES / HSTAGES: lists of STW(LR, PS) / STH(LR, PS) generated by the host; NB_ #defined at use
#define TWW(pi, d) (twW + ((pi) * 2 + (d)) * W)
#define TWH(pi, d) (twH + ((pi) * 2 + (d)) * H)
#define STW(LR, PS) NTT_STAGE(a_, W, NB_, LR, PS, TWW, dir_)
#define STH(LR, PS) NTT_STAGE(a_, H, NB_, LR, PS, TWH, dir_)

inline uint twN(__global const uint* twN1, __global const uint* twN2, uint pi, uint d, uint e) {
    return mm(twN1[(pi * 2 + d) * W + e % W], twN2[(pi * 2 + d) * H + e / W], PR(pi), PI(pi));
}

// ---------------- limb loading with lazy carries and the top fold
// limb j lives at pos (j % H) * W + j / H. Lazy carry XC[s][col] goes to the first limb of the next segment.
inline ulong lazyc(__global const uint* XC, uint j) {
    const uint h = j % H, col = j / H;
    if (h % SEG != 0) return 0;
    if (h > 0) return XC[(h / SEG - 1) * NCOL + col];
    if (col > 0) return XC[(NSEG - 1) * NCOL + col - 1];
    return 0;
}
inline ulong limb(__global const uint* X, __global const uint* XC, uint j) {
    return (ulong)X[(j % H) * W + j / H] + lazyc(XC, j);
}
// word q (0..4) of K * T, T = sum_i limb(WN + i) 2^(32 i), i < 5
inline uint ktword(__global const uint* X, __global const uint* XC, uint q) {
    ulong tv[5];
    for (uint i = 0; i < 5; ++i) tv[i] = limb(X, XC, WN + i);
    // normalise T into 32-bit words (T < 2^162)
    uint tw[6];
    ulong c = 0;
    for (uint i = 0; i < 5; ++i) { ulong v = tv[i] + c; tw[i] = (uint)v; c = v >> 32; }
    tw[5] = (uint)c;
    // K * T with K = KLO + KHI 2^32
    ulong acc = 0;
    uint out = 0;
    ulong carry = 0;
    for (uint w = 0; w <= q; ++w) {
        // word w = sum_{i} lo/hi parts: KLO*tw[w] (lo) + KLO*tw[w-1] (hi) + KHI*tw[w-1] (lo) + KHI*tw[w-2] (hi)
        ulong s = carry;
        ulong a0 = (ulong)KLO * tw[w];
        s += (uint)a0;
        if (w >= 1) { ulong a1 = (ulong)KLO * tw[w - 1]; s += a1 >> 32; ulong b1 = (ulong)KHI * tw[w - 1]; s += (uint)b1; }
        if (w >= 2) { ulong b2 = (ulong)KHI * tw[w - 2]; s += b2 >> 32; }
        out = (uint)s;
        carry = s >> 32;
    }
    return out;
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void fwdW(__global const uint* X, __global const uint* XC, __global uint* A, __global const uint* twW,
          __global const uint* twN1, __global const uint* twN2) {
    __local uint s1[3 * W];
    const uint h = get_group_id(0), t = get_local_id(0);
    for (uint w = t; w < W; w += WG) {
        const uint j = h + H * w;
        ulong v = 0;
        if (j < WN) {
            v = limb(X, XC, j);
            if (j < 5) v += ktword(X, XC, j);
        }
        s1[w] = (uint)(v % P0);
        s1[W + w] = (uint)(v % P1);
        s1[2 * W + w] = (uint)(v % P2);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    {
#define NB_ 3
        __local uint* a_ = s1;
        const uint dir_ = 0;
        WSTAGES
#undef NB_
    }
    for (uint i = t; i < 3 * W; i += WG) {
        const uint pi = i / W, kw = i % W;
        A[pi * LL + h * W + kw] = mm(s1[i], twN(twN1, twN2, pi, 0, h * kw), PR(pi), PI(pi));
    }
}

// mode 0 square, 1 forward only, 2 multiply by Bf
inline __attribute__((always_inline)) void tailImpl(__local uint* s1, __global uint* A, __global const uint* Bf,
                                                    __global const uint* twH, const int mode) {
    const uint g = get_group_id(0), t = get_local_id(0);
    // arrays: arr = pi * TG + c ; column kw = g*TG + c
    for (uint idx = t; idx < 3 * TG * H; idx += WG) {
        const uint c = idx % TG, rest = idx / TG, kh = rest % H, pi = rest / H;
        s1[(pi * TG + c) * H + kh] = A[pi * LL + kh * W + g * TG + c];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    {
#define NB_ (3 * TG)
        __local uint* a_ = s1;
        const uint dir_ = 0;
        HSTAGES
#undef NB_
    }
    if (mode == 1) {
        for (uint idx = t; idx < 3 * TG * H; idx += WG) {
            const uint c = idx % TG, rest = idx / TG, kh = rest % H, pi = rest / H;
            A[pi * LL + kh * W + g * TG + c] = s1[(pi * TG + c) * H + kh];
        }
        return;
    }
    for (uint idx = t; idx < 3 * TG * H; idx += WG) {
        const uint pi = idx / (TG * H);
        const uint p = PR(pi), pinv = PI(pi);
        uint v = s1[idx];
        if (mode == 0) v = mm(v, v, p, pinv);
        else {
            const uint c = (idx / H) % TG, kh = idx % H;
            v = mm(v, Bf[pi * LL + kh * W + g * TG + c], p, pinv);
        }
        s1[idx] = v;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    {
#define NB_ (3 * TG)
        __local uint* a_ = s1;
        const uint dir_ = 1;
        HSTAGES
#undef NB_
    }
    for (uint idx = t; idx < 3 * TG * H; idx += WG) {
        const uint c = idx % TG, rest = idx / TG, kh = rest % H, pi = rest / H;
        A[pi * LL + kh * W + g * TG + c] = s1[(pi * TG + c) * H + kh];
    }
}
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void tail(__global uint* A, __global const uint* twH) {
    __local uint s1[3 * TG * H];
    tailImpl(s1, A, A, twH, 0);
}
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void tailFwd(__global uint* A, __global const uint* twH) {
    __local uint s1[3 * TG * H];
    tailImpl(s1, A, A, twH, 1);
}
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void tailMul(__global uint* A, __global const uint* twH, __global const uint* Bf) {
    __local uint s1[3 * TG * H];
    tailImpl(s1, A, Bf, twH, 2);
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void invW(__global const uint* A, __global uint* Z, __global const uint* twW, __global const uint* twN1,
          __global const uint* twN2, const uint mulc) {
    __local uint s1[3 * W];
    const uint h = get_group_id(0), t = get_local_id(0);
    for (uint i = t; i < 3 * W; i += WG) {
        const uint pi = i / W, kw = i % W;
        s1[i] = mm(A[pi * LL + h * W + kw], twN(twN1, twN2, pi, 1, h * kw), PR(pi), PI(pi));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    {
#define NB_ 3
        __local uint* a_ = s1;
        const uint dir_ = 1;
        WSTAGES
#undef NB_
    }
    for (uint w = t; w < W; w += WG) {
        const uint m = h + H * w;
        if (m > 2 * WN - 2) continue;
        // scale (includes 1/L and the Montgomery factors) -> plain residues
        const uint r1 = mm(s1[w], S0, P0, PINV0);
        const uint r2 = mm(s1[W + w], S1, P1, PINV1);
        const uint r3 = mm(s1[2 * W + w], S2, P2, PINV2);
        // Garner: z = r1 + p1 t2 + p1 p2 t3
        const uint t2 = mm(sb(r2, r1, P1), C12M, P1, PINV1);  // (r2 - r1) / p1 mod p2
        uint x3 = mm(sb(r3, r1, P2), C13M, P2, PINV2);         // (r3 - r1) / p1 mod p3
        const uint t3 = mm(sb(x3, t2, P2), C23M, P2, PINV2);   // ((r3 - r1)/p1 - t2) / p2 mod p3
        const ulong pt2 = (ulong)P0 * t2;
        const ulong a = (ulong)t3 * P01LO, b = (ulong)t3 * P01HI;
        ulong s = (ulong)r1 + (uint)pt2 + (uint)a;
        const uint w0 = (uint)s;
        s = (s >> 32) + (pt2 >> 32) + (a >> 32) + (uint)b;
        const uint w1 = (uint)s;
        const uint w2 = (uint)((s >> 32) + (b >> 32));
        // times mulc
        ulong q0 = (ulong)w0 * mulc;
        ulong q1 = (ulong)w1 * mulc + (q0 >> 32);
        ulong q2 = (ulong)w2 * mulc + (q1 >> 32);
        const uint pos = h * W + w;
        Z[pos] = (uint)q0;
        Z[LL + pos] = (uint)q1;
        Z[2 * LL + pos] = (uint)q2;
    }
}

// ---------------- carry
inline void zget(__global const uint* Z, uint k, uint* z) {
    if (k > 2 * WN - 2) { z[0] = z[1] = z[2] = 0; return; }
    const uint pos = (k % H) * W + k / H;
    z[0] = Z[pos]; z[1] = Z[LL + pos]; z[2] = Z[2 * LL + pos];
}
// y_j = z_j + K z_(j+WN) as 4 words (j < WN), else 0
inline void yget(__global const uint* Z, uint j, uint* y) {
    y[0] = y[1] = y[2] = y[3] = 0;
    if (j >= WN) return;
    uint z[3], u[3];
    zget(Z, j, z);
    zget(Z, j + WN, u);
    // K*u, K = KLO + KHI 2^32, u < 2^85
    ulong a0 = (ulong)KLO * u[0], a1 = (ulong)KLO * u[1], a2 = (ulong)KLO * u[2];
    ulong b0 = (ulong)KHI * u[0], b1 = (ulong)KHI * u[1], b2 = (ulong)KHI * u[2];
    ulong s = (ulong)z[0] + (uint)a0;
    y[0] = (uint)s;
    s = (s >> 32) + z[1] + (a0 >> 32) + (uint)a1 + (uint)b0;
    y[1] = (uint)s;
    s = (s >> 32) + z[2] + (a1 >> 32) + (uint)a2 + (b0 >> 32) + (uint)b1;
    y[2] = (uint)s;
    s = (s >> 32) + (a2 >> 32) + (b1 >> 32) + (uint)b2;
    y[3] = (uint)s;
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void carry(__global const uint* Z, __global uint* X, __global uint* XC) {
    const uint gid = get_global_id(0);
    if (gid >= NSEG * NCOL) return;
    const uint col = gid % NCOL, s = gid / NCOL;
    const uint j0 = col * H + s * SEG;
    if (j0 > WN + 4) { XC[s * NCOL + col] = 0; return; }
    uint y1[4], y2[4], y3[4], y0[4];
    // preload y_(j0-3), y_(j0-2), y_(j0-1)
    if (j0 >= 3) yget(Z, j0 - 3, y3); else { y3[0] = y3[1] = y3[2] = y3[3] = 0; }
    if (j0 >= 2) yget(Z, j0 - 2, y2); else { y2[0] = y2[1] = y2[2] = y2[3] = 0; }
    if (j0 >= 1) yget(Z, j0 - 1, y1); else { y1[0] = y1[1] = y1[2] = y1[3] = 0; }
    ulong c = 0;
    for (uint r = 0; r < SEG; ++r) {
        const uint j = j0 + r;
        yget(Z, j, y0);
        ulong v = c + (ulong)y0[0] + y1[1] + y2[2] + y3[3];
        X[(j % H) * W + j / H] = (uint)v;
        c = v >> 32;
        for (int q = 0; q < 4; ++q) { y3[q] = y2[q]; y2[q] = y1[q]; y1[q] = y0[q]; }
    }
    XC[s * NCOL + col] = (uint)c;
}
