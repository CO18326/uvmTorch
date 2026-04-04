#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <exception>
using namespace std;
extern "C"{



int prefetch_memory(unsigned long ptr_val, unsigned long size_in_bytes, int device_id) {
    void* ptr = (void*)ptr_val;
   // cudaStream_t stream1;
    cudaError_t err; 
   cudaMemLocation location;
   location.type=cudaMemLocationTypeDevice;
     struct cudaPointerAttributes attributes;
    err = cudaPointerGetAttributes(&attributes, ptr);
  // err = cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
   // cudaStreamSynchronize(stream);
   if(attributes.type != cudaMemoryTypeManaged){
    return 0;
   } 
   
   if(device_id==1){
    location.id=0;
    
    try{
    err = cudaMemPrefetchAsync(ptr, size_in_bytes,location,0);

    }
        catch (const exception& e) {
        // print the exception
        cout << "Exception " << e.what() << endl;
    }

     }

    else{
   location.id=-1;
   err = cudaMemPrefetchAsync(ptr, size_in_bytes,location,0);}
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemPrefetchAsync failed: %s\n", cudaGetErrorString(err));
    } else {
    // printf("error");
    }

	return 0;
}

int prefetch_memory_batch(unsigned long ptr_val_arr, unsigned long size_in_bytes_arr, unsigned long count, int device_id, cudaStream_t stream) {
    void** ptr_arr = (void**)ptr_val_arr;
    unsigned long* size_arr = (unsigned long*)size_in_bytes_arr;
    unsigned long location_indx[]={0};
   // cudaStream_t stream1;
    cudaError_t err; 
   cudaMemLocation location;
   location.type=cudaMemLocationTypeDevice;
  // err = cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
   // cudaStreamSynchronize(stream);
    if(device_id==1){
    location.id=0;
    //err = cudaMemPrefetchAsync(ptr, size_in_bytes,location,0,stream);
    cudaMemLocation locations[]={location};

    err=cudaMemPrefetchBatchAsync(ptr_arr,size_arr,count,locations,location_indx,1,0,stream);
    


   }
   /** else{
   location.id=-1;
   err = cudaMemPrefetchAsync(ptr, size_in_bytes,location,0);}
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemPrefetchAsync failed: %s\n", cudaGetErrorString(err));
    } else {
    // printf("error");
    }*/

	return 0;
}


int prefetch_memory_cpu(unsigned long ptr_val, unsigned long size_in_bytes, int device_id, cudaStream_t stream) {
    void* ptr = (void*)ptr_val;
    //cudaStream_t stream1;
    cudaError_t err; 
    cudaMemLocation location;
   location.type=cudaMemLocationTypeDevice;
   // err = cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
   // cudaStreamSynchronize(stream);
   location.id=-1;
    err = cudaMemPrefetchAsync(ptr, size_in_bytes,location,0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemPrefetchAsync failed: %s\n", cudaGetErrorString(err));
    } else {
    // printf("error");
    }

        return 0;
}


int pin_memory_hint(unsigned long ptr_val, size_t size, int device) {
    cudaMemLocation location;
   location.type=cudaMemLocationTypeDevice;
    location.id=-1;

   void* ptr = (void*)ptr_val;
    cudaError_t err;


    // Set preferred location for the memory
if(device==0){    
err = cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, location);}
else{
location.id=0;
err = cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, location);


}
location.id=0;
cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, location);
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


