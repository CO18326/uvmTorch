#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

typedef int CUresult;

typedef CUresult (*cuGetProcAddress_t)(
    const char* symbol,
    void** pfn,
    int driverVersion,
    unsigned int flags);

extern "C"
CUresult cuGetProcAddress(
    const char* symbol,
    void** pfn,
    int driverVersion,
    unsigned int flags)
{
    static cuGetProcAddress_t real_func = nullptr;

    if (!real_func)
        real_func = (cuGetProcAddress_t)dlsym(RTLD_NEXT, "cuGetProcAddress");

    CUresult res = real_func(symbol, pfn, driverVersion, flags);

    if (strcmp(symbol, "cuLaunchKernel") == 0)
    {
        printf("🔥 cuLaunchKernel requested via cuGetProcAddress\n");
    }

    if (strcmp(symbol, "cuGraphLaunch") == 0)
    {
        printf("🔥 cuGraphLaunch requested via cuGetProcAddress\n");
    }

    return res;
}
