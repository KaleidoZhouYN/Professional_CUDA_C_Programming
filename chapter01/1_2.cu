/* 1.2. 从hello.cu中移除cudaDeviceReset函数，然后编译运行，看看会发生什么*/

#include <stdio.h>

__global__ void helloFromGPU(void)
{
    printf("Hello World from GPU!\n");
}

int main(void)
{
    // hello from cpu
    printf("Hello World form CPU!\n");

    helloFromGPU<<<1, 10>>>(); 
    // cudaDeviceReset(); 
    return 0; 
}

/* the output is :
Hello World form CPU!
*/

/* the reason is :
cudaDeviceReset 函数的作用是重置当前设备上的所有状态和资源，
以便在重新使用该设备之前清理和释放所有占用的资源。

具体而言，cudaDeviceReset 函数会清空当前设备上的所有 CUDA 上下文，
并且释放该设备上的所有内存和资源。这个函数通常在程序执行结束时被调用，
以确保在退出应用程序之前释放设备资源。

使用 cudaDeviceReset 函数的主要原因是在使用 CUDA 设备时，
可能会出现各种错误和异常情况，例如内存泄漏、未释放资源等。
这些问题可能会导致应用程序出现不可预测的行为，例如崩溃或不正确的输出结果。
通过在程序执行结束时调用 cudaDeviceReset 函数，可以确保释放所有设备资源，
从而避免这些问题的发生。

需要注意的是，cudaDeviceReset 函数将清空当前设备上的所有状态和资源，
包括任何未完成的 CUDA 操作和任何未释放的内存。
因此，应该谨慎使用此函数，以免丢失任何重要的数据或状态。
通常，只有在程序执行结束时才需要调用该函数，或者在切换到使用另一个设备之前调用该函数。
*/