# 习题答案

2.3 参考文件sumMatrixOnGPU-2D-grid-2D-block.cu，并将它用于整数矩阵的加法运算中，获取最佳的执行配置<br>
|dimx|dimy|-|-|-|-|-|-|
|:------|:-|:-|:-|:-|:-|:-|:-|
|- |2|4|8|16|32|64|128|
|2 |0.099|0.047|0.023|0.013|0.010|0.010|0.0098|
|4 |0.047|0.023|0.012|0.0060|0.0047|0.0059|0.0054|
|8 |0.023|0.011|0.0060|0.0042|0.0040|0.0042|0.0043|
|16|0.013|0.0067|0.0042|0.0041|0.0041|0.0042|<font color="#FF0000">0.000024</font>|
|32|0.0060|0.0043|0.0042|0.0041|0.0040|<font color="#FF0000">0.000021|<font color="#FF0000">0.000025|
64|0.0043|0.0043|0.0042|0.0041|<font color="#FF0000">0.000029|<font color="#FF0000">0.000026|<font color="#FF0000">0.000026|
128|0.0042|0.0041|0.0042|<font color="#FF0000">0.000025|<font color="#FF0000">0.000025|<font color="#FF0000">0.000025|<font color="#FF0000">0.000025|


2.4 参考文件sumMatrixOnGPU-2D-grid-1D-block.cu，新家一个内核，使得每个线程处理两个元素，获取最佳的执行配置<br>
|dimx|2|4|8|16|32|64|128|256|512|1024
|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|
|elapse|0.1008|0.0475|0.0237|0.0119|0.0059|0.0051|0.0052|0.0051|0.0049|0.0050|


2.5 借助程序checkDeviceInfo.cu，找到你的系统所支持的网格和块的最大尺寸<br>

Detected 4 CUDA Capable device(s) <br>
device 0: "Tesla V100-DGXS-32GB"<br>
CUDA Driver Version / Runtime Version           11.4 / 10.2 <br>
CUDA Capability Major/Minor version number:     7.0<br>
Total amount of global memory:                  31.74 MBytes (34078457856 bytes)<br>
Maximum sizes of each dimension of a block:     1024 x 1024 x 64 <br>
Maximum sizes of each dimensino of a grid:      2147483647 x 65535 x 6553