#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <stdio.h>
#include <unistd.h>

#define N (1 << 24)   // ~64 MB
#define ITERATIONS 10

__global__ void compute_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        for (int i = 0; i < 1000; i++) {
            x = x * 1.000001f + 0.000001f;
        }
        data[idx] = x;
    }
}

// Simple CPU busy loop
void cpu_compute(int ms) {
    nvtxRangePushA("CPU Compute");
    usleep(ms * 1000);
    nvtxRangePop();
}

int main() {
    float *h_data, *d_data;
    size_t bytes = N * sizeof(float);

    h_data = (float *)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    cudaMalloc(&d_data, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < ITERATIONS; i++) {

        nvtxRangePushA("Iteration");

        // CPU work overlaps GPU
        cpu_compute(20);

        nvtxRangePushA("HtoD memcpy");
        cudaMemcpyAsync(d_data, h_data, bytes,
                        cudaMemcpyHostToDevice, stream);
        nvtxRangePop();

        nvtxRangePushA("GPU Compute");
        compute_kernel<<<(N + 255) / 256, 256, 0, stream>>>(d_data, N);
        nvtxRangePop();

        nvtxRangePushA("DtoH memcpy");
        cudaMemcpyAsync(h_data, d_data, bytes,
                        cudaMemcpyDeviceToHost, stream);
        nvtxRangePop();

        nvtxRangePop(); // Iteration
    }

    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    cudaFree(d_data);
    free(h_data);

    return 0;
}
