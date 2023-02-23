/* 1.3. 用cudaDeviceSynchronize函数来替换hello.cu中的cudaDeviceReset函数，然后编译运行，看看会发生什么*/

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
    cudaDeviceSynchronize();
    return 0; 
}

/* the output is :
Hello World form CPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
Hello World from GPU!
*/

/*cudaDeviceSynchronize 函数是 CUDA API 中的一种同步函数，
其作用是等待当前设备上的所有任务都执行完毕，并确保设备执行完成之后，主机代码才继续执行。

在 CUDA 中，当主机代码启动设备上的任务时，这些任务通常是异步执行的，
即主机代码在任务完成之前不会等待设备上的任务。因此，如果主机代码需要使用设备上的结果，
或者需要在设备上启动新的任务，就需要确保设备上的任务已经执行完毕。

cudaDeviceSynchronize 函数就是用来解决这个问题的。调用该函数会使主机代码阻塞，
直到当前设备上的所有任务都执行完毕，并且设备已经完成执行。
这样，主机代码就可以安全地使用设备上的结果，或者启动新的任务，
而不必担心未完成的任务会导致不可预测的行为。

需要注意的是，cudaDeviceSynchronize 函数只会等待当前设备上的任务完成，
而不会等待其他设备上的任务完成。如果程序中使用了多个设备，
则需要在每个设备上分别调用该函数，以确保所有设备上的任务都已经完成。

另外，由于 cudaDeviceSynchronize 函数会阻塞主机代码的执行，
因此在实际应用中应该尽可能避免频繁调用该函数，以提高程序的执行效率。
*/
