#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \ 
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d , reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
} \

double cpuSecond()
{
    struct timeval tp; 
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void initialData(float *ip,int size)
{
    // generate ifferent seed for random number
    time_t t; 
    srand((unsigned) time(&t));

    for (int i =0; i<size; i++) {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f; 
    }
}


void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8; 
    bool match = 1; 
    for (int i = 0; i <N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i],i);
            break; 
        }
    }

    if (match) printf("Arrays match. \n\n");
}


void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A; 
    float *ib = B; 
    float *ic = C; 

    for (int iy =0; iy<ny; iy++) {
        for (int ix= 0; ix<nx; ix++) {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx; ib += nx; ic += nx; 
    }
}

__global__ void sumMatrixOnGPUMix(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x; 
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y; 
    unsigned int idx = iy*nx + ix; 

    if (ix < nx && iy < ny) {
        MatC[2*idx] = MatA[2*idx] + MatB[2*idx];
        MatC[2*idx+1] = MatA[2*idx+1] + MatB[2*idx+1];
    }
}


int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up to device
    int dev = 0; 
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up date size of matrix
    int nx = 1 << 13; 
    int ny = 1 << 14; 

    int nxy = 2*nx * ny; 
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    double iStart = cpuSecond();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = cpuSecond() - iStart; 

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    iStart = cpuSecond();
    sumMatrixOnHost (h_A, h_B, hostRef, 2*nx, ny);
    iElaps = cpuSecond() - iStart; 

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC; 
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    // invoke kernel at host side
    int dims[10] = {1<<1, 1<<2, 1<<3, 1<<4,1<<5,1<<6,1<<7,1<<8,1<<9,1<<10};
    for (int i = 0; i < 10; i++){
        int dimx = dims[i]; 
        int dimy = 1; 
        dim3 block(dimx, dimy);
        dim3 grid((nx+block.x-1)/(block.x), (ny+block.y-1)/block.y);

        iStart = cpuSecond();
        sumMatrixOnGPUMix <<< grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
        cudaDeviceSynchronize();
        iElaps = cpuSecond() - iStart; 
        printf("sumMatrixOnGPU2D<<<(%d,%d),(%d,%d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

        // copy kernel result back to host side
        cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

        // check device result
        checkResult(hostRef, gpuRef, nxy);
    }


    // free device global memory
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    cudaDeviceReset();

    return 0; 
}