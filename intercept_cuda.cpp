#define _GNU_SOURCE
#include <dlfcn.h>
#include <cuda.h>
#include <stdio.h>

typedef CUresult (*cuLaunchKernel_t)(
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
    void **extra);

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
    static cuLaunchKernel_t real_func = nullptr;

    if (!real_func)
        real_func = (cuLaunchKernel_t)dlsym(RTLD_NEXT, "cuLaunchKernel");

    printf("🔥 Kernel launch intercepted!\n");
    printf("Grid: %u %u %u\n", gridDimX, gridDimY, gridDimZ);
    printf("Block: %u %u %u\n", blockDimX, blockDimY, blockDimZ);

    return real_func(f,
                     gridDimX, gridDimY, gridDimZ,
                     blockDimX, blockDimY, blockDimZ,
                     sharedMemBytes, hStream,
                     kernelParams, extra);
}
