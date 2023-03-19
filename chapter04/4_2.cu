#include <cuda_runtime.h>
#include <stdio.h>

__device__ float devData; 

__global__ void checkGlobalVariable() {
    // display the original value
    printf("Device: the value of the global variable is %f\n", devData);

    // alter the values
    devData += 2.0f; 
}

int main(void) {
    // initialize the global variable
    float value = 3.14f; 

    float* devPtr; 
    cudaGetSymbolAddress((void**)&devPtr,devData);
    cudaMemcpy(devPtr, &value, sizeof(float) , cudaMemcpyHostToDevice);
    printf("Host:   cpoied %f to the global variable\n", value);

    // invoke the kernel
    checkGlobalVariable<<<1,1>>>();

    // copy the global variable back to the host
    cudaMemcpy(&value, devPtr, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Host:   the value changed by the kernel to %f\n", value);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}