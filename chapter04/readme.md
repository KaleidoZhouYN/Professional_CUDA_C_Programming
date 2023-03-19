# 习题答案
4.1 参考文件gloalVariable.cu。静态声明一个大小为5的全局浮点数组，用相同的值3.14初始化该全局数组。修改核函数，令每个线程都用相同的线程索引更改数组中的元素值。将该值与线程索引相乘。用5个线程调用核函数。

4.2 参考文件globalVariable.cu。使用数据传输函数cudaMemcpy()替换下列符号拷贝函数：cudaMemcpyToSymbol()、cudaMemcpyFromSymbol()。需要使用cudaGetSymbolAddress()获取全局变量地址。

