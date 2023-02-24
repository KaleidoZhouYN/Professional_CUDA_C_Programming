#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    int deviceCount = 0; 
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        printf("cudaGetDevieCount returned %d\n-> %s\n",
        (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0)
    {
        printf("There are no available devie(s) that support CUDA\n");
    }
    else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0; 

    dev = 0; 
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp; 
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("device %d: \"%s\"\n", dev, deviceProp.name);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA Driver Version / Runtime Version           %d.%d / %d.%d\n",
        driverVersion/1000, (driverVersion%100)/10,
        runtimeVersion/1000, (runtimeVersion%100)/10);
    printf("CUDA Capability Major/Minor version number:     %d.%d\n",
        deviceProp.major, deviceProp.minor);
    printf("Total amount of global memory:                  %.2f MBytes (%llu bytes)\n",
        (float)deviceProp.totalGlobalMem/(pow(1024.0,3)),
        (unsigned long long) deviceProp.totalGlobalMem);


    printf("Maximum sizes of each dimension of a block:     %d x %d x %d\n",
        deviceProp.maxThreadsDim[0],
        deviceProp.maxThreadsDim[1],
        deviceProp.maxThreadsDim[2]);

    printf("Maximum sizes of each dimensino of a grid:      %d x %d x %d\n",
        deviceProp.maxGridSize[0],
        deviceProp.maxGridSize[1],
        deviceProp.maxGridSize[2]);
    exit(1);
}