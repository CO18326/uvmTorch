#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>

/* ===== DRIVER API TYPES ===== */
typedef int CUresult;
typedef void* CUfunction;
typedef void* CUstream;

/* ===== RUNTIME API TYPES ===== */
typedef int cudaError_t;
typedef void* cudaStream_t;

/* ===============================
   INTERCEPT cudaLaunchKernel
   =============================== */

struct dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
};



extern "C"
cudaError_t cudaLaunchKernel(
    const void* func,
    dim3 gridDim,
    dim3 blockDim,
    void** args,
    size_t sharedMem,
    cudaStream_t stream)
{
    typedef cudaError_t (*real_t)(
        const void*, dim3, dim3, void**, size_t, cudaStream_t);

    static real_t real_func = nullptr;

    if (!real_func)
        real_func = (real_t)dlsym(RTLD_NEXT, "cudaLaunchKernel");

    printf("🔥 RUNTIME kernel launch intercepted\n");

    return real_func(func, gridDim, blockDim, args, sharedMem, stream);
}

/* ===============================
   INTERCEPT cuLaunchKernel
   =============================== */
extern "C"
CUresult cuLaunchKernel(
    CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,
    void **extra)
{
    typedef CUresult (*real_t)(
        CUfunction,
        unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int,
        unsigned int, CUstream, void**, void**);

    static real_t real_func = nullptr;

    if (!real_func)
        real_func = (real_t)dlsym(RTLD_NEXT, "cuLaunchKernel");

    printf("🔥 DRIVER kernel launch intercepted\n");

    return real_func(f,
                     gridDimX, gridDimY, gridDimZ,
                     blockDimX, blockDimY, blockDimZ,
                     sharedMemBytes, hStream,
                     kernelParams, extra);
}




extern "C"
cudaError_t cudaLaunchKernel_ptsz(
    const void* func,
    dim3 gridDim,
    dim3 blockDim,
    void** args,
    size_t sharedMem,
    cudaStream_t stream)
{
    static auto real_func =
        (decltype(&cudaLaunchKernel_ptsz))
        dlsym(RTLD_NEXT, "cudaLaunchKernel_ptsz");

    printf("🔥 cudaLaunchKernel_ptsz intercepted\n");

    return real_func(func, gridDim, blockDim, args, sharedMem, stream);
}
