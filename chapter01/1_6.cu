/*1.6. 为执行核函数的每个线程提供一个唯一个线程ID，通过内置变量threadIdx.x可以在内核
中对线程进行访问。在hello.cu中修改核函数的线程索引，使输出如下：
$ ./hello
Hello World from CPU!
Hello World form GPU thread 5!
*/

#include <stdio.h>

__global__ void helloFromGPU(void)
{
    if (threadIdx.x == 5)
        printf("Hello World from GPU thread %d!\n",threadIdx.x);
}

int main(void)
{
    // hello from cpu
    printf("Hello World form CPU!\n");

    helloFromGPU<<<1,10>>>(); 
    cudaDeviceReset(); 
    return 0; 
}

/*
the output is:
Hello World from CPU!
Hello World form GPU thread 5!
*/