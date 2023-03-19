#include <cuda_runtime.h>
#include<stdio.h>

void sumArraysOnHost(float *a, float*b, float*c, int nElem, int offset) {
    for (int i = 0; i < nElem; i++) {
        unsigned int k = i + offset; 
        if (k < nElem) c[i] = a[k] + b[k];
    }
}

__global__ void sumArrays(float* a, float* b, float* c, int nElem, int offset) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned int k = idx + offset; 
    if (k >= nElem)
        return; 

    c[idx] = a[k] + b[k];
}

__global__ void sumArraysZeroCopy(float* a, float* b, float* c, int nElem, int offset) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned int k = idx + offset; 
    if (k >= nElem)
        return; 

    c[idx] = a[k] + b[k];
}

void initialData(float* a, int nElem) {
    memset(a, 1, sizeof(nElem));
}

void checkResult(float* hostRef, float* gpuRef, int nElem) {
    bool equal = 1;
    for (int i = 0; i < nElem; i++)
        if (hostRef[i] != gpuRef[i]) {
            equal = 0; 
            break; 
        }
    printf("Host equal to gpu, %d",equal);
}

int main(int argc, char** argv) {
    // part 0 : set up device and array
    // set up device
    int dev = 0; 
    cudaSetDevice(dev);

    // get device properties
    cudaDeviceProp deviceProp; 
    cudaGetDeviceProperties(&deviceProp, dev);

    // check if support mapped memory
    if (!deviceProp.canMapHostMemory) {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
    printf("Using Device %d: %s ", dev, deviceProp.name);

    // set up date size of vectors
    int ipower = 10; 
    if (argc>1) ipower = atoi(argv[1]);
    int nElem = 1<<ipower; 
    size_t nBytes = nElem * sizeof(float);
    if (ipower < 18) {
        printf("Vector size %d power %d nbytes %3.0f KB\n", nElem, \
            ipower, (float)nBytes/(1024.f));
    }
    else {
        printf("Vector size %d power %d nbytes %3.0f MB\n", nElem, \
            ipower, (float)nBytes/(1024.0f*1024.f));
    }

    int offset = 0; 
    if (argc > 2) offset = atoi(argv[2]);

    // part 1: using device memory
    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef; 
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem, offset);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // set up execution configuration
    int iLen = 512;
    dim3 block (iLen);
    dim3 grid ((nElem+block.x-1)/block.x);

    // invoke kernel at host side
    sumArrays <<<grid, block>>>(d_A, d_B, d_C, nElem, offset);

    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    // part 2: using zerocopy memory for array A and B
    // allocate zerocpy memory
    unsigned int flags = cudaHostAllocMapped;
    cudaHostAlloc((void **)&h_A, nBytes, flags);
    cudaHostAlloc((void **)&h_B, nBytes, flags);

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // pass the pointer to device
    cudaHostGetDevicePointer((void **)&d_A, (void *)h_A, 0);
    cudaHostGetDevicePointer((void **)&d_B, (void *)h_B, 0);

    // add at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem, offset);    

    // execute kernel with zero copy memory
    sumArraysZeroCopy <<<grid, block>>>(d_A, d_B, d_C, nElem, offset);

    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free memory
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    cudaDeviceReset();
    return EXIT_SUCCESS;
}