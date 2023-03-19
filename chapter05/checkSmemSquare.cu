
#include <cuda_runtime.h>
#include <stdio.h>

#define BDIMY 32
#define BDIMX 32


__global__ void warmup(int *out) {
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx; 

    // wait for all threads to complete
    __syncthreads();

    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadRow(int *out) {
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx; 

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int *out) {
    // static shared memory
    __shared__ int tile[BDIMX][BDIMY];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.x][threadIdx.y] = idx; 

    // wait for all thread to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadCol(int *out) {
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory sotre operation
    tile[threadIdx.y][threadIdx.x] = idx; 

    // wait for all threads to complete
    __syncthreads(); 

    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setColReadRow(int *out) {
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory sotre operation
    tile[threadIdx.x][threadIdx.y] = idx; 

    // wait for all threads to complete
    __syncthreads(); 

    // shared memory load operation
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadColDyn(int *out) {
    // dynamic shared memory
    extern __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;

    // shared memory store operation
    tile[row_idx] = row_idx; 

    // wait for all threads to complete
    __syncthreads(); 

    // shared memory load operation
    out[row_idx] = tile[col_idx];
}

__global__ void setRowReadColPad(int *out) {
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX+IPAD];

    // mapping from thread index to global memory offset
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x; 

    // shared memory sotre operation
    tile[threadIdx.y][threadIdx.x] = idx; 

    // wait for all threads to complete
    __syncthreads(); 

    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDynPad(int *out) {
    // dynamic shared memory
    extern __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x; 
    unsigned int col_idx = threadIdx.x * (blockDim.x + IPAD) + threadIdx.x; 

    unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x; 

    // shared memory store operation
    tile[row_idx] = g_idx; 

    // wait for all thread to complete
    __syncthreads(); 

    // shared memory load operation
    out[g_idx] = tile[col_idx];
}

__device__ __managed__ int* devPtr; 

int main() {
    // set up device
    int dev = 0; 
    cudaSetDevice(dev);

    // get device properties
    cudaDeviceProp deviceProp; 
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("Using Device %d: %s ", dev, deviceProp.name);

    // set blockdim and grid im
    dim3 block(BDIMX,BDIMY);
    dim3 grid(1,1);

    // use unified memory
    cudaMallocManaged((void**)&devPtr, BDIMX*BDIMY);

    // warmup 
    warmup<<<grid,block>>>(devPtr);
    cudaDeviceSynchronize();

    // row row
    setRowReadRow<<<grid,block>>>(devPtr);
    cudaDeviceSynchronize();

    // col col
    setColReadCol<<<grid,block>>>(devPtr);
    cudaDeviceSynchronize();

    // row col
    setRowReadCol<<<grid,block>>>(devPtr);
    cudaDeviceSynchronize();

    // col row
    setColReadRow<<<grid,block>>>(devPtr);
    cudaDeviceSynchronize();

    // free managed memory
    cudaFree(devPtr);
    
    // reset device
    cudaDeviceReset();
    return EXIT_SUCCESS;
}