// gpu_backend.h: the small GPU interface hyprp needs, implemented with OpenCL (any GPU) and, when compiled with
// HYPRP_CUDA, with the CUDA driver API + NVRTC (NVIDIA). The kernel source is the same (hyprp.cl, written so it
// compiles as OpenCL C and as CUDA C++); the CUDA build defines HYPRP_CUDA when compiling it with NVRTC.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "logline.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#ifdef HYPRP_CUDA
#include <cuda.h>
#include <nvrtc.h>
#endif

struct GpuBuf { uint64_t h = 0; bool operator==(const GpuBuf& o) const { return h == o.h; } };
struct GpuKernel { int id = -1; };

class Gpu {
public:
    virtual ~Gpu() {}
    std::string name, backend;
    bool fp64Fast = false;  // FP64 at half the FP32 rate (P100, V100, A100, H100, B200): hyprp's FP64 mode pays off
    // the card's UUID as nvidia-smi writes it ("GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"); "" when the driver gives
    // none. It names the physical card: on Kaggle or Colab each session gets another one.
    std::string uuid;
    // compiles source (the backend adds its own defines); returns false and fills log on error
    virtual bool build(const std::string& src, std::string& log) = 0;
    virtual GpuKernel kernel(const char* name) = 0;
    virtual GpuBuf alloc(size_t bytes, const void* init) = 0;
    virtual void write(GpuBuf b, size_t off, size_t bytes, const void* src) = 0;
    virtual void read(GpuBuf b, size_t bytes, void* dst) = 0;
    virtual void copy(GpuBuf dst, GpuBuf src, size_t bytes) = 0;
    virtual void setArg(GpuKernel k, int idx, GpuBuf b) = 0;
    virtual void setArg(GpuKernel k, int idx, int32_t v) = 0;
    virtual void setArg(GpuKernel k, int idx, uint32_t v) = 0;
    // groups x wg work-items; with profAcc the kernel's time (ms) is added to *profAcc (see flushProf)
    virtual void launch(GpuKernel k, size_t groups, size_t wg, double* profAcc) = 0;
    virtual void flushProf() = 0;  // add pending kernel times (ms) to their accumulators
    virtual void throttle() = 0;   // keep the queue from growing without bound
    virtual void finish() = 0;
};

// 16 UUID bytes -> "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" (nvidia-smi's form)
inline std::string uuidText(const unsigned char* b) {
    char s[48];
    std::snprintf(s, sizeof(s), "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0], b[1],
                  b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
    return s;
}

#define GCHECK(e, what)                                                                   \
    do {                                                                                    \
        long long _e = (long long)(e);                                                      \
        if (_e != 0) { logFatal("%s error %lld at %s:%d", what, _e, __FILE__, __LINE__); std::exit(1); } \
    } while (0)

// ------------------------------------------------------------------ OpenCL
class GpuCL : public Gpu {
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    cl_device_id dev = nullptr;
    cl_program prog = nullptr;
    std::vector<cl_kernel> ks;
    bool prof;
    std::vector<std::pair<cl_event, double*>> pend;
    cl_event evPrev = nullptr;
    long long count = 0;

public:
    GpuCL(int devIdx, bool profiling) : prof(profiling) {
        backend = "OpenCL";
        cl_uint np = 0;
        GCHECK(clGetPlatformIDs(0, nullptr, &np), "OpenCL");
        std::vector<cl_platform_id> plats(np);
        GCHECK(clGetPlatformIDs(np, plats.data(), nullptr), "OpenCL");
        std::vector<cl_device_id> devs;
        for (auto p : plats) {
            cl_uint nd = 0;
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS) continue;
            std::vector<cl_device_id> d(nd);
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, d.data(), nullptr);
            devs.insert(devs.end(), d.begin(), d.end());
        }
        if (devs.empty()) { logFatal("no OpenCL GPU"); std::exit(1); }
        dev = devs[std::min<size_t>((size_t)devIdx, devs.size() - 1)];
        char nm[256] = "", ver[256] = "", drv[256] = "";
        cl_uint cus = 1, clk = 0;
        cl_ulong lmem = 0, gmem = 0;
        clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(nm), nm, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(ver), ver, nullptr);
        clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof(drv), drv, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clk), &clk, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmem), &lmem, nullptr);
        clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, nullptr);
        nm[sizeof(nm) - 1] = ver[sizeof(ver) - 1] = drv[sizeof(drv) - 1] = '\0';  // in case a string was cut off
        name = nm;
        unsigned char ub[16];  // cl_khr_device_uuid: CL_DEVICE_UUID_KHR = 0x106A, 16 bytes
        if (clGetDeviceInfo(dev, 0x106A, 16, ub, nullptr) == CL_SUCCESS) uuid = uuidText(ub);
        logInfo("device: %s | %s | driver %s | %u CU @ %u MHz | local %llu KB | global %llu MB | OpenCL | uuid %s", nm,
                ver, drv, cus, clk, (unsigned long long)lmem / 1024, (unsigned long long)gmem >> 20,
                uuid.empty() ? "unknown" : uuid.c_str());
        cl_int err;
        ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
        GCHECK(err, "OpenCL");
        q = clCreateCommandQueue(ctx, dev, prof ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
        GCHECK(err, "OpenCL");
    }
    bool build(const std::string& src, std::string& log) override {
        const char* s = src.c_str();
        cl_int err;
        prog = clCreateProgramWithSource(ctx, 1, &s, nullptr, &err);
        GCHECK(err, "OpenCL");
        if (clBuildProgram(prog, 1, &dev, "", nullptr, nullptr) != CL_SUCCESS) {
            size_t len = 0;
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
            log.assign(len, 0);
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, len, &log[0], nullptr);
            return false;
        }
        return true;
    }
    GpuKernel kernel(const char* nm) override {
        cl_int err;
        ks.push_back(clCreateKernel(prog, nm, &err));
        GCHECK(err, "OpenCL kernel");
        return GpuKernel{(int)ks.size() - 1};
    }
    GpuBuf alloc(size_t bytes, const void* init) override {
        cl_int err;
        cl_mem b = clCreateBuffer(ctx, CL_MEM_READ_WRITE | (init ? CL_MEM_COPY_HOST_PTR : 0), bytes, (void*)init, &err);
        GCHECK(err, "OpenCL alloc");
        GpuBuf g;
        g.h = (uint64_t)(uintptr_t)b;
        return g;
    }
    static cl_mem M(GpuBuf b) { return (cl_mem)(uintptr_t)b.h; }
    void write(GpuBuf b, size_t off, size_t bytes, const void* src) override {
        GCHECK(clEnqueueWriteBuffer(q, M(b), CL_TRUE, off, bytes, src, 0, nullptr, nullptr), "OpenCL write");
    }
    void read(GpuBuf b, size_t bytes, void* dst) override {
        GCHECK(clEnqueueReadBuffer(q, M(b), CL_TRUE, 0, bytes, dst, 0, nullptr, nullptr), "OpenCL read");
    }
    void copy(GpuBuf dst, GpuBuf src, size_t bytes) override {
        GCHECK(clEnqueueCopyBuffer(q, M(src), M(dst), 0, 0, bytes, 0, nullptr, nullptr), "OpenCL copy");
    }
    void setArg(GpuKernel k, int idx, GpuBuf b) override { cl_mem m = M(b); clSetKernelArg(ks[k.id], idx, sizeof(cl_mem), &m); }
    void setArg(GpuKernel k, int idx, int32_t v) override { clSetKernelArg(ks[k.id], idx, 4, &v); }
    void setArg(GpuKernel k, int idx, uint32_t v) override { clSetKernelArg(ks[k.id], idx, 4, &v); }
    void launch(GpuKernel k, size_t groups, size_t wg, double* acc) override {
        size_t g = groups * wg;
        cl_event e;
        GCHECK(clEnqueueNDRangeKernel(q, ks[k.id], 1, nullptr, &g, &wg, 0, nullptr, acc ? &e : nullptr), "OpenCL launch");
        if (acc) {
            pend.push_back({e, acc});
            if (pend.size() >= 600) flushProf();
        }
    }
    void flushProf() override {
        if (pend.empty()) return;
        clWaitForEvents(1, &pend.back().first);
        for (auto& pe : pend) {
            cl_ulong a, b;
            clGetEventProfilingInfo(pe.first, CL_PROFILING_COMMAND_START, 8, &a, nullptr);
            clGetEventProfilingInfo(pe.first, CL_PROFILING_COMMAND_END, 8, &b, nullptr);
            *pe.second += (b - a) * 1e-6;
            clReleaseEvent(pe.first);
        }
        pend.clear();
    }
    void throttle() override {
        if ((++count & 255) == 0) {
            cl_event ev;
            GCHECK(clEnqueueMarker(q, &ev), "OpenCL marker");
            clFlush(q);
            if (evPrev) { clWaitForEvents(1, &evPrev); clReleaseEvent(evPrev); }
            evPrev = ev;
        }
    }
    void finish() override { clFinish(q); }
};

#ifdef HYPRP_CUDA
// ------------------------------------------------------------------ CUDA (driver API + NVRTC)
class GpuCUDA : public Gpu {
    CUdevice dev = 0;
    CUcontext ctx = nullptr;
    CUmodule mod = nullptr;
    int ccMajor = 0, ccMinor = 0;
    struct K { CUfunction f; std::vector<uint64_t> args; };
    std::vector<K> ks;
    bool prof;
    struct P { CUevent a, b; double* acc; };
    std::vector<P> pend;
    CUevent evPrev = nullptr;
    long long count = 0;

public:
    static bool available() {
        if (std::getenv("HYPRP_BACKEND") && std::string(std::getenv("HYPRP_BACKEND")) == "opencl") return false;
        if (cuInit(0) != CUDA_SUCCESS) return false;
        int n = 0;
        return cuDeviceGetCount(&n) == CUDA_SUCCESS && n > 0;
    }
    GpuCUDA(int devIdx, bool profiling) : prof(profiling) {
        backend = "CUDA";
        int n = 0;
        GCHECK(cuDeviceGetCount(&n), "CUDA");
        GCHECK(cuDeviceGet(&dev, std::min(devIdx, n - 1)), "CUDA");
        char nm[256];
        GCHECK(cuDeviceGetName(nm, sizeof(nm), dev), "CUDA");
        int sms = 1, drv = 0;
        cuDeviceGetAttribute(&ccMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
        cuDeviceGetAttribute(&ccMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
        cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
        cuDriverGetVersion(&drv);
        size_t mem = 0;
        cuDeviceTotalMem(&mem, dev);
        name = nm;
        // half-rate FP64: P100 (6.0), V100 (7.0), A100 (8.0), H100 (9.0), B200 (10.0); the others run it at 1/32-1/64
        // (compute 12.0 is consumer/workstation Blackwell, RTX 50 and RTX PRO 6000)
        const int cc = ccMajor * 10 + ccMinor;
        fp64Fast = cc == 60 || cc == 70 || cc == 80 || cc == 90 || cc == 100;
        CUuuid cu;
        if (cuDeviceGetUuid(&cu, dev) == CUDA_SUCCESS) uuid = uuidText((const unsigned char*)cu.bytes);
        logInfo("device: %s | compute %d.%d | CUDA driver %d | %d SM | global %llu MB | CUDA%s | uuid %s", nm, ccMajor,
                ccMinor, drv, sms, (unsigned long long)mem >> 20, fp64Fast ? " | full-rate FP64" : "",
                uuid.empty() ? "unknown" : uuid.c_str());
        GCHECK(cuDevicePrimaryCtxRetain(&ctx, dev), "CUDA");
        GCHECK(cuCtxSetCurrent(ctx), "CUDA");
    }
    bool build(const std::string& src, std::string& log) override {
        std::string full = "#define HYPRP_CUDA 1\n" + src;
        nvrtcProgram p;
        GCHECK(nvrtcCreateProgram(&p, full.c_str(), "hyprp.cu", 0, nullptr, nullptr), "NVRTC");
        const std::string arch = "--gpu-architecture=compute_" + std::to_string(ccMajor * 10 + ccMinor);
        const char* o[2] = {arch.c_str(), "--std=c++17"};
        nvrtcResult r = nvrtcCompileProgram(p, 2, o);
        size_t ls = 0;
        nvrtcGetProgramLogSize(p, &ls);
        log.assign(ls, 0);
        if (ls) nvrtcGetProgramLog(p, &log[0]);
        if (r != NVRTC_SUCCESS) { nvrtcDestroyProgram(&p); return false; }
        size_t ps = 0;
        GCHECK(nvrtcGetPTXSize(p, &ps), "NVRTC");
        std::string ptx(ps, 0);
        GCHECK(nvrtcGetPTX(p, &ptx[0]), "NVRTC");
        nvrtcDestroyProgram(&p);
        if (mod) cuModuleUnload(mod);
        ks.clear();
        CUresult lr = cuModuleLoadDataEx(&mod, ptx.c_str(), 0, nullptr, nullptr);
        if (lr != CUDA_SUCCESS) {
            const char* es = nullptr;
            cuGetErrorString(lr, &es);
            log += std::string("\ncuModuleLoadDataEx failed: ") + (es ? es : "?");
            return false;
        }
        return true;
    }
    GpuKernel kernel(const char* nm) override {
        K k;
        GCHECK(cuModuleGetFunction(&k.f, mod, nm), "CUDA kernel");
        k.args.assign(16, 0);
        ks.push_back(k);
        return GpuKernel{(int)ks.size() - 1};
    }
    GpuBuf alloc(size_t bytes, const void* init) override {
        CUdeviceptr p;
        GCHECK(cuMemAlloc(&p, bytes), "CUDA alloc");
        if (init) GCHECK(cuMemcpyHtoD(p, init, bytes), "CUDA write");
        else GCHECK(cuMemsetD8(p, 0, bytes), "CUDA memset");
        GpuBuf g;
        g.h = (uint64_t)p;
        return g;
    }
    void write(GpuBuf b, size_t off, size_t bytes, const void* src) override {
        GCHECK(cuMemcpyHtoD((CUdeviceptr)b.h + off, src, bytes), "CUDA write");
    }
    void read(GpuBuf b, size_t bytes, void* dst) override { GCHECK(cuMemcpyDtoH(dst, (CUdeviceptr)b.h, bytes), "CUDA read"); }
    void copy(GpuBuf dst, GpuBuf src, size_t bytes) override {
        GCHECK(cuMemcpyDtoD((CUdeviceptr)dst.h, (CUdeviceptr)src.h, bytes), "CUDA copy");
    }
    void setArg(GpuKernel k, int idx, GpuBuf b) override { ks[k.id].args[idx] = b.h; }
    void setArg(GpuKernel k, int idx, int32_t v) override { uint64_t x = 0; std::memcpy(&x, &v, 4); ks[k.id].args[idx] = x; }
    void setArg(GpuKernel k, int idx, uint32_t v) override { uint64_t x = 0; std::memcpy(&x, &v, 4); ks[k.id].args[idx] = x; }
    void launch(GpuKernel k, size_t groups, size_t wg, double* acc) override {
        K& kk = ks[k.id];
        void* params[16];
        for (int i = 0; i < 16; ++i) params[i] = &kk.args[i];  // scalars are read from the low bytes (little endian)
        P pe{nullptr, nullptr, acc};
        if (acc) {
            cuEventCreate(&pe.a, CU_EVENT_DEFAULT);
            cuEventCreate(&pe.b, CU_EVENT_DEFAULT);
            cuEventRecord(pe.a, 0);
        }
        GCHECK(cuLaunchKernel(kk.f, (unsigned)groups, 1, 1, (unsigned)wg, 1, 1, 0, 0, params, nullptr), "CUDA launch");
        if (acc) {
            cuEventRecord(pe.b, 0);
            pend.push_back(pe);
            if (pend.size() >= 600) flushProf();
        }
    }
    void flushProf() override {
        if (pend.empty()) return;
        cuEventSynchronize(pend.back().b);
        for (auto& pe : pend) {
            float ms = 0;
            cuEventElapsedTime(&ms, pe.a, pe.b);
            *pe.acc += ms;
            cuEventDestroy(pe.a);
            cuEventDestroy(pe.b);
        }
        pend.clear();
    }
    void throttle() override {
        if ((++count & 255) == 0) {
            CUevent ev;
            cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING);
            cuEventRecord(ev, 0);
            if (evPrev) { cuEventSynchronize(evPrev); cuEventDestroy(evPrev); }
            evPrev = ev;
        }
    }
    void finish() override { GCHECK(cuCtxSynchronize(), "CUDA sync"); }
};
#endif

inline std::unique_ptr<Gpu> makeGpu(int devIdx, bool prof) {
#ifdef HYPRP_CUDA
    if (GpuCUDA::available()) return std::unique_ptr<Gpu>(new GpuCUDA(devIdx, prof));
#endif
    return std::unique_ptr<Gpu>(new GpuCL(devIdx, prof));
}
