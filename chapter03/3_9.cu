__global__ void nestedHelloWorld(int const iSize, int iDepth, int iMaxDepth) {
    int tid = threadIdx.x; 
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    printf("Recursion=%d: Hello World from thread %d"
        "block %d\n",iDepth,tid,blockIdx.x);

    // condition to stop recursivce execution
    if (iSize == 1 || iDepth >= iMaxDepth) return;

    // reduce block size to half
    int nthreads = iSize >> 1; 

    // thread 0 launches child grid recursively
    if (idx == 0 && nthreads > 0) {
        nestedHelloWorld<<<1, nthreads>>>(nthreas,++iDepth);
        printf("------> nested execution depth: %d\n",iDepth);
    }
}