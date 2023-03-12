# 习题

3.1当在CUDA中展开循环、数据块或线程束时，可以提高性能的两个主要原因是什么？解释每种展开是如何提升指令吞吐量的。<br>
A：两个主要原因分别是：
1.减少指令消耗 2.隐藏内存延迟<br>
展开循环减少了逻辑判断指令<br>
展开数据库隐藏了内存延迟<br>
展开线程束避免了循环控制和线程同步逻辑<br>

3.2参考核函数reduceUnrolling8实现核函数reduceUnrolling16, 在这个函数中每个线程处理16个数据块。将该函数的性能与reduceUnrolling8内核性能进行比较，通过nvprof使用合适的指标与事件来解释性能差异。


3.3参考核函数reduceUnrolling8,替换以下的代码段：

`int a1 = g_idata[idx];`<br>
`int a2 = g_idata[idx];`<br>
`int a3 = g_idata[idx];`<br>
`int a4 = g_idata[idx];`<br>
`int b1 = g_idata[idx];`<br>
`int b2 = g_idata[idx];`<br>
`int b3 = g_idata[idx];`<br>
`int b4 = g_idata[idx];`<br>
`g_idata[idx] = a1+a2+a3+a4+b1+b2+b3+b4;`<br>
使用下面再功能上等价的代码：

`int *ptr = g_idata + idx;`<br>
`int tmp = 0;` <br>

`// Increment tmp 8 times with values strided by blockDim.x`<br>
`for (int i = 0; i < 8; i++) { `<br>
`   temp += *ptr; ptr += blockDim.x; `<br>
`}`<br>
`g_idata[idx] = tmp;`

比较每次的性能并解释使用nvprof指标的差异。

-----------

3.4参考核函数reduceCompleteUnrollWarps8。不要将vmem声明为volatile修饰符，而是使用__syncthreads。注意__syncthreads必须被线程块里的所有线程调用。比较两个核函数的性能并使用nvprof来解释所有的差异。

---------

3.5用C语言实现浮点数s的求和归约。

-----------

3.6参考核函数reduceInterleaved和reduceCompleteUnrollWarps8,实现每个浮点数s的版本。比较它们的性能，选择合适的指标与/或事件来解释所有差距。它们相比于操作整数数据类型有什么不同的吗？

A:因为一个CUDA核心中有不同数量的ALU和FPU，FPU能够处理int类型，但是ALU无法处理浮点类型

-------

3.7被动态地产生孩子的全局数据进行更改，这种改变什么时候能保证对其父亲可见？

A：子网格结束时进行同步之后，可以保证对其父亲可见

---------

3.8参考文件nestedHelloWorld.cu，用图3-30所示的方法实现一个新的核函数。

--------

3.9参考文件nestedHelloWorld.cy，实现一个新的核函数，使其可以用给定深度来限制嵌套层。
