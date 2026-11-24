// micro-benchmark: memory bandwidth (copy) and FP64 FMA throughput
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cstdio>
#include <vector>
static const char* src = R"(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void copy(__global const double2* a, __global double2* b) { size_t i = get_global_id(0); b[i] = a[i]; }
__kernel void fma64(__global double* o, double s) {
  double x = get_global_id(0) * 1e-9, y = 1.0 + x, z = 0.5 + x, w = 0.25 + x;
  for (int i = 0; i < 1024; ++i) { x = fma(x, s, 0.1); y = fma(y, s, 0.2); z = fma(z, s, 0.3); w = fma(w, s, 0.4); }
  o[get_global_id(0)] = x + y + z + w;
}
__kernel void lds(__global double2* o) {
  __local double2 s[512];
  uint t = get_local_id(0);
  double2 v = (double2)(t, 1);
  for (int i = 0; i < 256; ++i) { s[(t * 8 + i) & 511] = v; barrier(CLK_LOCAL_MEM_FENCE); v += s[(t * 3 + i) & 511]; barrier(CLK_LOCAL_MEM_FENCE); }
  o[get_global_id(0)] = v;
}
)";
int main() {
    cl_platform_id p; clGetPlatformIDs(1, &p, nullptr);
    cl_device_id d; clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, nullptr);
    cl_int e;
    cl_context c = clCreateContext(nullptr, 1, &d, nullptr, nullptr, &e);
    cl_command_queue q = clCreateCommandQueue(c, d, CL_QUEUE_PROFILING_ENABLE, &e);
    cl_program pr = clCreateProgramWithSource(c, 1, &src, nullptr, &e);
    if (clBuildProgram(pr, 1, &d, "", nullptr, nullptr)) { std::puts("build fail"); return 1; }
    auto time = [&](cl_kernel k, size_t g, size_t l) {
        cl_event ev; double best = 1e9;
        for (int r = 0; r < 20; ++r) {
            clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, &l, 0, nullptr, &ev);
            clWaitForEvents(1, &ev);
            cl_ulong a, b;
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, 8, &a, nullptr);
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, 8, &b, nullptr);
            clReleaseEvent(ev);
            if ((b - a) * 1e-9 < best) best = (b - a) * 1e-9;
        }
        return best;
    };
    for (size_t mb : {2, 4, 8, 64, 256}) {
        size_t n = mb * 1048576 / 16;
        cl_mem a = clCreateBuffer(c, CL_MEM_READ_WRITE, n * 16, nullptr, &e);
        cl_mem b = clCreateBuffer(c, CL_MEM_READ_WRITE, n * 16, nullptr, &e);
        cl_kernel k = clCreateKernel(pr, "copy", &e);
        clSetKernelArg(k, 0, sizeof(a), &a); clSetKernelArg(k, 1, sizeof(b), &b);
        double t = time(k, n, 64);
        std::printf("copy %3zu MB: %.4f ms  %.1f GB/s (read+write)\n", mb, t * 1e3, 2.0 * n * 16 / t / 1e9);
        clReleaseMemObject(a); clReleaseMemObject(b);
    }
    {
        size_t n = 1 << 20;
        cl_mem o = clCreateBuffer(c, CL_MEM_READ_WRITE, n * 8, nullptr, &e);
        cl_kernel k = clCreateKernel(pr, "fma64", &e);
        double s = 0.999999;
        clSetKernelArg(k, 0, sizeof(o), &o); clSetKernelArg(k, 1, 8, &s);
        double t = time(k, n, 64);
        std::printf("fp64 fma: %.1f GFLOPS\n", n * 1024.0 * 4 * 2 / t / 1e9);
    }
    {
        size_t n = 64 * 1024;
        cl_mem o = clCreateBuffer(c, CL_MEM_READ_WRITE, n * 16, nullptr, &e);
        cl_kernel k = clCreateKernel(pr, "lds", &e);
        clSetKernelArg(k, 0, sizeof(o), &o);
        double t = time(k, n, 64);
        std::printf("lds round trips: %.1f G double2 write+read per s, %.2f us per WG-iteration pair\n",
                    n * 256.0 / t / 1e9, t / 256 * 1e6);
    }
    return 0;
}
