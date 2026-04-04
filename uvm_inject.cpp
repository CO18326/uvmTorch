#include <cupti.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

static void CUPTIAPI callback(
    void *userdata,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const CUpti_CallbackData *cbInfo)
{
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
        cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 &&
        cbInfo->callbackSite == CUPTI_API_ENTER)
    {
        printf("🔥 Before kernel launch → injecting UVM prefetch\n");

        // Example: prefetch a known pointer
        // Replace with your tracked UVM pointer list
        void* ptr = userdata;

        if (ptr != NULL)
        {
            int device;
            cudaGetDevice(&device);

            cudaMemPrefetchAsync(ptr, 1024 * 1024, device, 0);
        }
    }
}

extern "C"
void __attribute__((constructor)) init()
{
    CUpti_SubscriberHandle subscriber;

    cuptiSubscribe(&subscriber,
                   (CUpti_CallbackFunc)callback,
                   NULL);

    cuptiEnableCallback(1,
                        subscriber,
                        CUPTI_CB_DOMAIN_RUNTIME_API,
                        CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
}
