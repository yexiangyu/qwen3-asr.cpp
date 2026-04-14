#include "qwen3asr_c_api.h"
#include <stdio.h>

int main() {
    printf("=== Qwen3ASR C API Test ===\n\n");
    
    printf("Version: %s\n", qwen3asr_get_version());
    
    int n_devices = qwen3_get_device_count();
    printf("Available devices: %d\n", n_devices);
    
    for (int i = 0; i < n_devices; i++) {
        qwen3_device_info info;
        if (qwen3_get_device_info(i, &info) == 0) {
            printf("  [%d] %s (%s) - free: %zu MB, total: %zu MB\n",
                   i, info.name, info.description,
                   info.memory_free / (1024*1024),
                   info.memory_total / (1024*1024));
            qwen3_free_device_info(&info);
        }
    }
    
    printf("\nTesting ASR handle...\n");
    qwen3asr_handle handle;
    if (qwen3asr_init(&handle) != 0) {
        printf("Init failed: %s\n", qwen3_get_last_error());
        return 1;
    }
    
    printf("Device: %s\n", qwen3asr_get_device_name(handle));
    
    qwen3asr_free(handle);
    printf("\nDone.\n");
    
    return 0;
}