#include <cuda_runtime.h>
#include<stdio.h>

int main(int argc, char** argv)
{
    // define grid and block structure
    int nElem = 1024; 
    
    const unsigned int block_x_size[4] = {1024, 512, 256, 128};

    for (int i = 0; i < 4; i++) {
        dim3 block (block_x_size[i]);
        dim3 grid ((nElem+block.x-1)/block.x);
        printf("grid.x %d block.x %d \n", grid.x, block.x);
    }

    // reset device before you leave
    cudaDeviceReset();
    
    return(0);
}