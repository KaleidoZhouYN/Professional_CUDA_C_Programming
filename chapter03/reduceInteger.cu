#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define gen_reduceUnrollWarps(k) \
__global__ void reduceUnrollWarps##k(int *g_idata, int* g_odata, unsigned int n) { \
    unsigned int tid = threadIdx.x;                                               \ 
    unsigned int idx = blockIdx.x * blockDim.x * k + threadIdx.x;                \
    int *idata = g_idata + blockIdx.x*blockDim.x*k;                              \
    if (idx+(k-1)*blockDim.x < n) {                                              \
        float tmp = 0;                                                           \
        for (int i = 0; i < k; i++)                                              \
            tmp += g_idata[idx+i*blockDim.x];                                    \
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
        for (int stride = 32; stride >0; stride>>=1)                             \
            vmem[tid] += vmem[tid+stride];                                       \
    }                                                                            \
    if (tid == 0) g_odata[blockIdx.x] = idata[0];                                \
}

gen_reduceUnrollWarps(8);

#define run_reduce(s)                                               \
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);    \
    cudaDeviceSynchronize();                                        \
    iStart = seconds();                                             \
    s<<<grid, block>>>(d_idata, d_odata, size);                    \
    cudaDeviceSynchronize();                                        \
    iElaps = seconds() - iStart;                                    \
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost); \
    gpu_sum = 0;                                                    \
    for (int i=0; i<grid.x; i++)                                    \
        gpu_sum += h_odata[i];                                      \
    printf("gpu %s    elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n", \
            #s, iElaps, gpu_sum, grid.x, block.x); 


int recursiveReduce(int *data, int const size) {
    // terminate check
    if (size == 1) return data[0];

    // renew the stride
    int const stride = size / 2; 

    // in-place reduction
    for (int i = 0; i < stride; i++) {
        data[i] += data[i + stride];
    }

    // call recursively
    return recursiveReduce(data, stride);
}

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x; 

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x; 

    // boundary check
    if (tid + blockIdx.x * blockDim.x >= n) return; 

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within block
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboredLess(int * g_idata, int * g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x; 
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 


    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x; 

    // boundary check
    if (idx >= n) return; 

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // convert tid into local array index
        int index = 2 * stride * tid; 
        if (index < blockDim.x) {
            idata[index] += idata[index + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x; 
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x; 

    // boundary check
    if (idx >= n) return; 

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceUnrollWarps(int *g_idata, int *g_odata, unsigned int n, unsigned int k) {
    // set thread ID
    unsigned int tid = threadIdx.x; 
    unsigned int idx = blockIdx.x * blockDim.x*8 + threadIdx.x; 

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x*blockDim.x*k;                              

    // unrolling 8
    if (idx+(k-1)*blockDim.x < n) {                                              
        float tmp = 0;                                                           
        for (int i = 0; i < k; i++)                                              
            tmp += g_idata[idx+i*blockDim.x];                                    
        g_idata[idx] = tmp;                                                      
    }            
    __syncthreads();                                                             

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {               
        if (tid < stride) {                                                      
            idata[tid] += idata[tid + stride];                                   
        }                                 

        // synchronize within threadblock                                       
        __syncthreads();                                                         
    }                                                                            
    if (tid < 32) {                                                              
        volatile int *vmem = idata;                                              
        for (int stride = 32; stride >0; stride>>=1)                             
            vmem[tid] += vmem[tid+stride];                                       
    }                                    

    // write result for this block to global mem                                        
    if (tid == 0) g_odata[blockIdx.x] = idata[0];                                
}

int main(int argc, char **argv) {
    // set up device
    int dev = 0; 
    cudaDeviceProp deviceProp; 
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    cudaSetDevice(dev);

    bool bResult = false; 

    // initialization
    int size = 1<<24; // total number of elements to reduce
    printf("    with array size %d ", size);

    // execution configuration
    int blocksize = 512; // initial block size
    if (argc > 1) {
        blocksize = atoi(argv[1]);
    }
    dim3 block(blocksize,1);
    dim3 grid((size+block.x-1)/block.x,1);
    printf("grid %d block %d\n",grid.x,block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++) {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (int)(rand() & 0xFF);
    }
    memcpy(tmp, h_idata, bytes);

    size_t iStart, iElaps;
    int gpu_sum = 0; 

    // allocate device memory
    int *d_idata = NULL; 
    int *d_odata = NULL; 
    cudaMalloc((void**) &d_idata, bytes);
    cudaMalloc((void**) &d_odata, grid.x*sizeof(int));

    // cpu reduction
    iStart = seconds();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce  elapsed %d ms cpu_sum: %d\n", iElaps,cpu_sum);

    //kernel 1: reduceNeighbored
    // run_reduce(warmup);
    
    // kernel 1: reduceNeighbored
    run_reduce(reduceNeighbored);

    // kernel 2 : reduceNeighboredLess
    run_reduce(reduceNeighboredLess);

    // kernel 3 : reduceInterleaved
    run_reduce(reduceInterleaved);

    // kernel 4 : reduceUnrollWarps8
    run_reduce(reduceUnrollWarps8);

    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x/8*sizeof(int), cudaMemcpyDeviceToHost);

    gpu_sum = 0; 
    for (int i = 0; i < grid.x / 8; i++)
        gpu_sum += h_odata[i];

    printf("gpu Cmptnroll  elapsed %d ms gpu_sum: %d <<<grid %d block %d>>>\n",
            iElaps, gpu_sum, grid.x, block.x);

    // free host memory 
    free(h_idata);
    free(h_odata);

    // free device memory 
    cudaFree(d_idata);
    cudaFree(d_odata);

    // reset device
    cudaDeviceReset();

    // check the results
    bResult = (gpu_sum == cpu_sum);
    if (!bResult) printf("Test failed!\n");
    return EXIT_SUCCESS;
}