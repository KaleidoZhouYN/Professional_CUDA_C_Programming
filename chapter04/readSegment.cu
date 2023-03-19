int main(int argc, char **argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    cudaSetDevice(dev);
    // set up array size
    int nElem = 1<<20; // total number of elements to reduce
    printf(" with array size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);
    // set up offset for summary
    int blocksize = 512;
    int offset = 0;
    if (argc>1) offset = atoi(argv[1]);
    if (argc>2) blocksize = atoi(argv[2]);
    // execution configuration
    dim3 block (blocksize,1);
    dim3 grid ((nElem+block.x-1)/block.x,1);
    // allocate host memory
    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef = (float *)malloc(nBytes);
    // initialize host array
    initialData(h_A, nElem);
    memcpy(h_B,h_A,nBytes);
    // summary at host side
    sumArraysOnHost(h_A, h_B, hostRef,nElem,offset);
    // allocate device memory
    float *d_A,*d_B,*d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);
    // copy data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_A, nBytes, cudaMemcpyHostToDevice);
    // kernel 1:
    double iStart = seconds();
    warmup <<< grid, block >>> (d_A, d_B, d_C, nElem, offset);
    cudaDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("warmup <<< %4d, %4d >>> offset %4d elapsed %f sec\n",
    grid.x, block.x,
    offset, iElaps);
    iStart = seconds();
    readOffset <<< grid, block >>> (d_A, d_B, d_C, nElem, offset);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("readOffset <<< %4d, %4d >>> offset %4d elapsed %f sec\n",
    grid.x, block.x,
    offset, iElaps);
    // copy kernel result back to host side and check device results
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nElem-offset);
    // copy kernel result back to host side and check device results
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nElem-offset);
    // copy kernel result back to host side and check device results
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nElem-offset);
    // free host and device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);

     // reset device
    cudaDeviceReset();
    return EXIT_SUCCESS;
}