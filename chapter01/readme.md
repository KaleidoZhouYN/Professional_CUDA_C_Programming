# 可能会遇到的问题
1. Max supported GCC version<br>

|CUDA version|max supported GCC version|
|:-----|:------| 
|12|	12.1|
|11.4.1+, 11.5, 11.6, 11.7, 11.8|	11|
|11.1, 11.2, 11.3, 11.4.0	|10 |
|11|	9|
|10.1, 10.2|	8|
|9.2, 10.0|	7|
|9.0, 9.1|	6|
|8	|5.3|
|7	|4.9|
|5.5, 6	|4.8|
|4.2, 5|	4.6|
|4.1|	4.5|
|4.0|	4.4|

if your default gcc version is higher than the max supported GCC version,here is the solution:<br>
as an example for nvcc-10.2, plese use 
`nvcc -ccbin /usr/bin/gcc-7` or `nvcc -ccbin /usr/bin/g++-7` while compiling your .cu file

