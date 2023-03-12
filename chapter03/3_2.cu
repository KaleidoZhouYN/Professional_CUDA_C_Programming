__global__ void reduceUnrollWarps16(int *g_idata, int *g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x*16 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x*blockDim.x*16;

    // unrolling 16
    if (idx + 15*blockDim.x < n) {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2*blockDim.x];
        int a4 = g_idata[idx + 3*blockDim.x];
        int a5 = g_idata[idx + 4*blockDim.x];
        int a6 = g_idata[idx + 5*blockDim.x];
        int a7 = g_idata[idx + 6*blockDim.x];
        int a8 = g_idata[idx + 7*blockDim.x];
        int a9 = g_idata[idx + 8*blockDim.x];
        int a10 = g_idata[idx + 9*blockDim.x];
        int a11 = g_idata[idx + 10*blockDim.x];
        int a12 = g_idata[idx + 11*blockDim.x];
        int a13 = g_idata[idx + 12*blockDim.x];
        int a14 = g_idata[idx + 13*blockDim.x];
        int a15 = g_idata[idx + 14*blockDim.x];
        int a16 = g_idata[idx + 15*blockDim.x];
        g_idata[idx] = a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12+a13+a14+a15+a16;
    }
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride>32; stride >>=1) {
        if (tid < stride) {
            idata[tid] += idata[tid+stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // unrolling warp
    if (tid < 32) {
        volatile int *vmem = idata; 
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] + vmem[tid+1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}