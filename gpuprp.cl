// GPU PRP prototype for N = 2^n - c (c small, positive).
// Zero-padded double-precision FFT squaring + int64 carry + folding with 2^n == c (mod N).
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

typedef double2 cplx;

inline cplx cmul(cplx a, cplx b) { return (cplx)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

__kernel void load(__global const long* x, __global cplx* buf, const uint L, const uint M) {
    uint j = get_global_id(0);
    if (j >= M) return;
    buf[j] = (cplx)(j < L ? (double)x[j] : 0.0, 0.0);
}

// Stockham radix-4 pass (Bainville formulation). t = M/4 threads. p = current sub-length.
// tw[m] = exp(-2*pi*i*m/M), m in [0, M).  dir = +1 forward, -1 inverse (conjugate twiddles).
__kernel void fft_r4(__global const cplx* x, __global cplx* y, __global const cplx* tw,
                     const uint p, const uint M, const int dir) {
    const uint t = M >> 2;
    const uint i = get_global_id(0);
    if (i >= t) return;
    const uint k = i & (p - 1);
    const uint j = ((i - k) << 2) + k;
    const uint step = M / (4 * p);  // alpha = -2pi k/(4p) -> table index k*step
    cplx w1 = tw[(k * step) & (M - 1)];
    cplx w2 = tw[(2 * k * step) & (M - 1)];
    cplx w3 = tw[(3 * k * step) & (M - 1)];
    if (dir < 0) { w1.y = -w1.y; w2.y = -w2.y; w3.y = -w3.y; }
    cplx u0 = x[i];
    cplx u1 = cmul(x[i + t], w1);
    cplx u2 = cmul(x[i + 2 * t], w2);
    cplx u3 = cmul(x[i + 3 * t], w3);
    cplx v0 = u0 + u2, v1 = u0 - u2, v2 = u1 + u3, d = u1 - u3;
    cplx v3 = dir > 0 ? (cplx)(d.y, -d.x) : (cplx)(-d.y, d.x);  // * (-i) forward, * (+i) inverse
    y[j] = v0 + v2;
    y[j + p] = v1 + v3;
    y[j + 2 * p] = v0 - v2;
    y[j + 3 * p] = v1 - v3;
}

__kernel void fft_r2(__global const cplx* x, __global cplx* y, __global const cplx* tw,
                     const uint p, const uint M, const int dir) {
    const uint t = M >> 1;
    const uint i = get_global_id(0);
    if (i >= t) return;
    const uint k = i & (p - 1);
    const uint j = ((i - k) << 1) + k;
    cplx w = tw[(k * (M / (2 * p))) & (M - 1)];
    if (dir < 0) w.y = -w.y;
    cplx u0 = x[i];
    cplx u1 = cmul(x[i + t], w);
    y[j] = u0 + u1;
    y[j + p] = u0 - u1;
}

__kernel void square(__global cplx* buf, const uint M) {
    uint j = get_global_id(0);
    if (j >= M) return;
    cplx a = buf[j];
    buf[j] = cmul(a, a);
}

// round convolution, optionally multiply by small constant, track max round-off error
__kernel void round_conv(__global const cplx* buf, __global long* conv, const uint M, const uint P,
                         const double invM, const long mulc, __global volatile uint* maxerr) {
    uint j = get_global_id(0);
    if (j >= P) return;
    long v = 0;
    if (j < M) {
        double r = buf[j].x * invM;
        double rr = rint(r);
        double e = fabs(r - rr);
        uint eb = (uint)(e * 4294967295.0);
        atomic_max(maxerr, eb);
        v = (long)rr * mulc;
    }
    conv[j] = v;
}

#define BLK 16
inline long bal(long v, uint w, long* carry) {
    // balanced digit in [-2^(w-1), 2^(w-1))
    const long base = 1L << w, hb = base >> 1, mask = base - 1;
    long d = ((v + hb) & mask) - hb;
    *carry = (v - d) >> w;
    return d;
}

__kernel void carryA(__global long* a, __global long* cout, const uint nblk, const uint w) {
    uint b = get_global_id(0);
    if (b >= nblk) return;
    long carry = 0;
    for (uint t = 0; t < BLK; ++t) {
        uint i = b * BLK + t;
        long c2;
        a[i] = bal(a[i] + carry, w, &c2);
        carry = c2;
    }
    cout[b] = carry;
}

__kernel void carryB(__global long* a, __global const long* cin, __global long* cout, const uint nblk,
                     const uint w) {
    uint b = get_global_id(0);
    if (b >= nblk) return;
    long carry = b > 0 ? cin[b - 1] : 0;
    for (uint t = 0; t < BLK && carry != 0; ++t) {
        uint i = b * BLK + t;
        long c2;
        a[i] = bal(a[i] + carry, w, &c2);
        carry = c2;
    }
    cout[b] = carry;
}

__kernel void check_zero(__global const long* c, const uint nblk, __global volatile uint* flag) {
    uint b = get_global_id(0);
    if (b >= nblk) return;
    if (c[b] != 0) atomic_or(flag, 1u);
}

// T = Lo + c * H, where value = sum D_i 2^(w i), n = q*w + r.
__kernel void fold(__global const long* D, __global long* T, const uint Dlen, const uint Tlen,
                   const uint q, const uint r, const uint w, const long cc) {
    uint j = get_global_id(0);
    if (j >= Tlen) return;
    long v = 0;
    if (r == 0) {
        if (j < q) v = D[j];
        if (q + j < Dlen) v += cc * D[q + j];
    } else {
        long dq = D[q];
        long a = dq & ((1L << r) - 1);
        long bq = (dq - a) >> r;
        if (j < q) v = D[j];
        else if (j == q) v = a;
        if (j == 0) v += cc * bq;
        if (q + 1 + j < Dlen) v += (cc << (w - r)) * D[q + 1 + j];
    }
    T[j] = v;
}
