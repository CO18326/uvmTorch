#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>

// Define the function pointer type for the real cudaLaunchKernel
typedef cudaError_t (*cudaLaunchKernel_t)(const void* func, dim3 gridDim, dim3 blockDim, 
                                          void** args, size_t sharedMem, cudaStream_t stream);

extern "C" cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, 
                                        void** args, size_t sharedMem, cudaStream_t stream) {
    // --- YOUR INJECTION CODE STARTS HERE ---
    // This runs on the CPU immediately before the kernel is queued.
    static int launch_count = 0;
    printf("[Injection] Launching kernel #%d (Grid: %dx%dx%d)\n", 
           ++launch_count, gridDim.x, gridDim.y, gridDim.z);
    
    // Example: Force synchronization for debugging (makes execution synchronous)
    // cudaDeviceSynchronize(); 
    // --- YOUR INJECTION CODE ENDS HERE ---

    // Find the 'real' cudaLaunchKernel in the CUDA library and call it
    static cudaLaunchKernel_t real_func = (cudaLaunchKernel_t)dlsym(RTLD_NEXT, "cudaLaunchKernel");
    return real_func(func, gridDim, blockDim, args, sharedMem, stream);
}
