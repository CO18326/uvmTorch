#include <cupti.h>
#include <cuda.h>
#include <stdio.h>

void CUPTIAPI callback(void *userdata,
                       CUpti_CallbackDomain domain,
                       CUpti_CallbackId cbid,
                       const CUpti_CallbackData *cbInfo)
{
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
        cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000)
    {
        printf("🔥 Kernel launch intercepted (runtime)\n");
    }

    if (domain == CUPTI_CB_DOMAIN_DRIVER_API &&
        cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)
    {
        printf("🔥 Kernel launch intercepted (driver)\n");
    }
}

extern "C" void init_cupti()
{
    CUpti_SubscriberHandle subscriber;
    cuptiSubscribe(&subscriber,
                   (CUpti_CallbackFunc)callback,
                   NULL);

    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
}
