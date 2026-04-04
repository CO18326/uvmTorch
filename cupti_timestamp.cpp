#include <torch/extension.h>
#include <cupti.h>
#include <stdexcept>

uint64_t get_cupti_timestamp() {
    uint64_t ts = 0;

    CUptiResult result = cuptiGetTimestamp(&ts);
    if (result != CUPTI_SUCCESS) {
        const char* errstr;
        cuptiGetResultString(result, &errstr);
        throw std::runtime_error(errstr);
    }

    return ts;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_timestamp", &get_cupti_timestamp,
          "Get CUPTI global GPU timestamp (ns)");
}
