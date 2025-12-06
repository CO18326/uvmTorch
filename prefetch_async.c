#include <cuda_runtime.h>
#include <stdio.h>


extern "C"{



int prefetch_memory(unsigned long ptr_val, unsigned long size_in_bytes, int device_id, cudaStream_t stream) {
    void* ptr = (void*)ptr_val;
   // cudaStream_t stream1;
    cudaError_t err; 
  // err = cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
   // cudaStreamSynchronize(stream);
    if(device_id==1){
    err = cudaMemPrefetchAsync(ptr, size_in_bytes,0,stream);}
    else{
   err = cudaMemPrefetchAsync(ptr, size_in_bytes,cudaCpuDeviceId);}
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemPrefetchAsync failed: %s\n", cudaGetErrorString(err));
    } else {
    // printf("error");
    }

	return 0;
}


int prefetch_memory_cpu(unsigned long ptr_val, unsigned long size_in_bytes, int device_id, cudaStream_t stream) {
    void* ptr = (void*)ptr_val;
    //cudaStream_t stream1;
    cudaError_t err; 
   // err = cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
   // cudaStreamSynchronize(stream);
    err = cudaMemPrefetchAsync(ptr, size_in_bytes,cudaCpuDeviceId);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemPrefetchAsync failed: %s\n", cudaGetErrorString(err));
    } else {
    // printf("error");
    }

        return 0;
}


int pin_memory_hint(unsigned long ptr_val, size_t size, int device) {
    

   void* ptr = (void*)ptr_val;
    cudaError_t err;


    // Set preferred location for the memory
if(device==0){    
err = cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);}
else{
err = cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, 1);


}
cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, 0);

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemAdviseSetPreferredLocation failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}



int print_first_byte() {
    //unsigned char* ptr = (unsigned char*)address;
    int* a;
    cudaMallocManaged(&a,sizeof(int));
   *a=1;
    //printf("Value at address %lx: 0x%02x\n", address, *ptr);
    return 0;
}

int cuda_malloc(unsigned long size_in_gb){

void* b;

cudaMalloc(&b,size_in_gb*(1024*1024*1024));

return 0;

}



}



