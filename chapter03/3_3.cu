#define gen_reduceUnrollWarps_useFor(k) \
_global__ void reduceUnrollWarps##kUseFor(int *g_idata, int* g_odata, unsigned int n) { \
    unsigned int tid = threadIdx.x;                                               \ 
    unsigned int idx = blockIdx.x * blockDim.x * k + threadIdx.x;                \
    int *idata = g_idata + blockIdx.x*blockDim.x*k;                              \
    if (idx+(k-1)*blockDim.x < n) {                                              \
        int *ptr = g_idata + idx;                                               \
        int tmp = 0;                                                           \
        for (int i = 0; i < k; i++) {                                             \
            tmp += *ptr;                                                            \
            ptr += blockDim.x;                                                  \
        }                                                                       \
        g_idata[idx] = tmp;                                                      \
    }                                                                            \
    __syncthreads();                                                             \
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {               \
        if (tid < stride) {                                                      \
            idata[tid] += idata[tid + stride];                                   \
        }                                                                        \
        __syncthreads();                                                         \
    }                                                                            \
    if (tid < 32) {                                                              \
        volatile int *vmem = idata;                                              \
        vmem[tid] += vmem[tid + 32];                                            \
        vmem[tid] += vmem[tid + 16];                                            \
        vmem[tid] += vmem[tid + 8];                                             \
        vmem[tid] += vmem[tid + 4];                                             \
        vmem[tid] += vmem[tid + 2];                                             \
        vmem[tid] + vmem[tid+1];                                                \
    }                                                                            \
    if (tid == 0) g_odata[blockIdx.x] = idata[0];                                \
}

gen_reduceUnrollWarps_useFor(8);