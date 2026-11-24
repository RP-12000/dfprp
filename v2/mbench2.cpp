// micro-benchmark: cost of a complex multiply-add chain in FP64, double-float (FP32 pairs),
// and 32-bit modular (Montgomery) arithmetic, as complex-mults per second.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cstdio>
static const char* src = R"(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define ITER 256
// ---------- FP64 complex mul + add
__kernel void c64(__global double2* o, double2 w) {
  double2 x = (double2)(get_global_id(0) * 1e-9, 1.0), y = (double2)(0.5, 0.25);
  for (int i = 0; i < ITER; ++i) {
    double2 t = (double2)(fma(x.x, w.x, -x.y * w.y), fma(x.x, w.y, x.y * w.x));
    x = y + t; y = y - t;
  }
  o[get_global_id(0)] = x + y;
}
// ---------- double-float: value = hi + lo
typedef float2 df;
inline df two_sum(float a, float b) { float s = a + b; float bb = s - a; float e = (a - (s - bb)) + (b - bb); return (df)(s, e); }
inline df fast_two_sum(float a, float b) { float s = a + b; return (df)(s, b - (s - a)); }
inline df df_add(df a, df b) { df s = two_sum(a.x, b.x); df t = two_sum(a.y, b.y); s.y += t.x; s = fast_two_sum(s.x, s.y); s.y += t.y; return fast_two_sum(s.x, s.y); }
inline df df_add_sloppy(df a, df b) { df s = two_sum(a.x, b.x); s.y += a.y + b.y; return fast_two_sum(s.x, s.y); }
inline df df_mul(df a, df b) { float p = a.x * b.x; float e = fma(a.x, b.x, -p); e = fma(a.x, b.y, fma(a.y, b.x, e)); return fast_two_sum(p, e); }
inline df df_neg(df a) { return (df)(-a.x, -a.y); }
typedef struct { df re, im; } cdf;
inline cdf cdf_mul(cdf a, cdf b) { cdf r; r.re = df_add_sloppy(df_mul(a.re, b.re), df_neg(df_mul(a.im, b.im))); r.im = df_add_sloppy(df_mul(a.re, b.im), df_mul(a.im, b.re)); return r; }
inline cdf cdf_add(cdf a, cdf b) { cdf r; r.re = df_add_sloppy(a.re, b.re); r.im = df_add_sloppy(a.im, b.im); return r; }
inline cdf cdf_sub(cdf a, cdf b) { cdf r; r.re = df_add_sloppy(a.re, df_neg(b.re)); r.im = df_add_sloppy(a.im, df_neg(b.im)); return r; }
__kernel void cdf_k(__global float4* o, float4 wv) {
  cdf w; w.re = wv.xy; w.im = wv.zw;
  cdf x; x.re = (df)(get_global_id(0) * 1e-9f, 0); x.im = (df)(1, 0);
  cdf y; y.re = (df)(0.5f, 0); y.im = (df)(0.25f, 0);
  for (int i = 0; i < ITER; ++i) { cdf t = cdf_mul(x, w); x = cdf_add(y, t); y = cdf_sub(y, t); }
  o[get_global_id(0)] = (float4)(x.re + y.re, x.im + y.im);
}
// ---------- 32-bit prime p = 15*2^27+1 (Montgomery), real mult + add/sub (an NTT butterfly)
#define P 2013265921u
#define PINV 2013265919u  // -p^-1 mod 2^32
inline uint mont(uint a, uint b) { ulong t = (ulong)a * b; uint m = (uint)t * PINV; ulong u = (t + (ulong)m * P) >> 32; uint r = (uint)u; return r >= P ? r - P : r; }
inline uint addm(uint a, uint b) { uint s = a + b; return s >= P ? s - P : s; }
inline uint subm(uint a, uint b) { return a >= b ? a - b : a + P - b; }
__kernel void ntt32(__global uint* o, uint w) {
  uint x = get_global_id(0) + 1, y = 12345;
  for (int i = 0; i < ITER; ++i) { uint t = mont(x, w); x = addm(y, t); y = subm(y, t); }
  o[get_global_id(0)] = x ^ y;
}
)";
int main() {
    cl_platform_id p; clGetPlatformIDs(1, &p, nullptr);
    cl_device_id d; clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, nullptr);
    cl_int e;
    cl_context c = clCreateContext(nullptr, 1, &d, nullptr, nullptr, &e);
    cl_command_queue q = clCreateCommandQueue(c, d, CL_QUEUE_PROFILING_ENABLE, &e);
    cl_program pr = clCreateProgramWithSource(c, 1, &src, nullptr, &e);
    if (clBuildProgram(pr, 1, &d, "", nullptr, nullptr)) {
        char log[8192]; clGetProgramBuildInfo(pr, d, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::puts(log); return 1;
    }
    const size_t n = 1 << 20, l = 64;
    cl_mem o = clCreateBuffer(c, CL_MEM_READ_WRITE, n * 16, nullptr, &e);
    auto run = [&](const char* name, const void* arg, size_t asz, double mults_per_item) {
        cl_kernel k = clCreateKernel(pr, name, &e);
        clSetKernelArg(k, 0, sizeof(o), &o); clSetKernelArg(k, 1, asz, arg);
        double best = 1e9;
        for (int r = 0; r < 10; ++r) {
            cl_event ev;
            clEnqueueNDRangeKernel(q, k, 1, nullptr, &n, &l, 0, nullptr, &ev);
            clWaitForEvents(1, &ev);
            cl_ulong a, b;
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, 8, &a, nullptr);
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, 8, &b, nullptr);
            clReleaseEvent(ev);
            if ((b - a) * 1e-9 < best) best = (b - a) * 1e-9;
        }
        std::printf("%-6s %8.2f G butterflies/s\n", name, n * mults_per_item / best / 1e9);
    };
    double w64[2] = {0.6, 0.8};
    float w32[4] = {0.6f, 1e-9f, 0.8f, -1e-9f};
    unsigned wi = 123456789u;
    run("c64", w64, sizeof(w64), 256);
    run("cdf_k", w32, sizeof(w32), 256);
    run("ntt32", &wi, sizeof(wi), 256);
    return 0;
}
