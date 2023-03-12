#define gen_reduceInterleaved_type(type)                                              \
__global__ void reduceInterleaved #type(type *g_idata, type *g_odata, unsigned int n) \
{                                                                                     \
    unsigned int tid = threadIdx.x;                                                   \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;                         \
    type *idata = g_idata + blockIdx.x * blockDim.x;                                  \
    if (idx >= n)                                                                     \
        return;                                                                       \
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)                       \
    {                                                                                 \
        if (tid < stride)                                                             \
        {                                                                             \
            idata[tid] += idata[tid + stride];                                        \
        }                                                                             \
        __syncthreads();                                                              \
    }                                                                                 \
    if (tid == 0)                                                                     \
        g_odata[blockIdx.x] = idata[0];                                               \
}

gen_reduceInterleaved_type(float);

#define gen_reduceCompleteUnrollWarps8_type(type)                                        \
__global__ void reduceCompleteUnrollWarps8(type *g_idata, type *g_odata, unsigned int n) \
{                                                                                        \
    unsigned int tid = threadIdx.x;                                                      \
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;                        \
    type *idata = g_idata + blockIdx.x * blockDim.x * k;                                 \
    if (idx + 7 * blockDim.x < n)                                                        \
    {                                                                                    \
        type a1 = g_idata[idx];                                                          \
        type a2 = g_idata[idx + blockDim.x];                                             \
        type a3 = g_idata[idx + 2 * blockDim.x];                                         \
        type a4 = g_idata[idx + 3 * blockDim.x];                                         \
        type a5 = g_idata[idx + 4 * blockDim.x];                                         \
        type a6 = g_idata[idx + 5 * blockDim.x];                                         \
        type a7 = g_idata[idx + 6 * blockDim.x];                                         \
        type a8 = g_idata[idx + 7 * blockDim.x];                                         \
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;                            \
    }                                                                                    \
    __syncthreads();                                                                     \
    if (blockDim.x >= 1024 && tid < 512)                                                 \
        idata[tid] += idata[tid + 512];                                                  \
    __syncthreads();                                                                     \
    if (blockDim.x >= 512 && tid < 256)                                                  \
        idata[tid] += idata[tid + 256];                                                  \
    __syncthreads();                                                                     \
    if (blockDim.x >= 256 && tid < 128)                                                  \
        idata[tid] += idata[tid + 128];                                                  \
    __syncthreads();                                                                     \
    if (blockDim.x >= 128 && tid < 64)                                                   \
        idata[tid] += idata[tid + 64];                                                   \
    __syncthreads();                                                                     \
    if (tid < 32)                                                                        \
    {                                                                                    \
        volatile type *vmem = idata;                                                     \
        vmem[tid] += vmem[tid + 32];                                                     \
        vmem[tid] += vmem[tid + 16];                                                     \
        vmem[tid] += vmem[tid + 8];                                                      \
        vmem[tid] += vmem[tid + 4];                                                      \
        vmem[tid] += vmem[tid + 2];                                                      \
        vmem[tid] + vmem[tid + 1];                                                       \
    }                                                                                    \
    if (tid == 0)                                                                        \
        g_odata[blockIdx.x] = idata[0];                                                  \
}

gen_reduceCompleteUnrollWarps8_type(float);