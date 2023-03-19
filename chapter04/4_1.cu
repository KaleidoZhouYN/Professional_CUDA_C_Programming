#include <cuda_runtime.h>
#include <stdio.h>

__device__ float devData[5];

__global__ void checkGlobalVariable() {
    devData[threadIdx.x] *= threadIdx.x;
}

int main(void)
{
    // initialize the global variable
    float value[5] = {3.14f,3.14f,3.14f,3.14f,3.14f}; 
    cudaMemcpyToSymbol(devData, &value, 5*sizeof(float));

    // invoke the kernel
    checkGlobalVariable<<<1,5>>>();

    // copy the global variable back to the host
    cudaMemcpyFromSymbol(&value, devData, 5*sizeof(float));
    for (int i =0; i < 5; i++)
        printf("Host:   the value changed by the kernel to %f\n", value[i]);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}