# mPimPy: A Software-Hardware Co-Design Simulator for Memristor-Based In-Memory Computing

## *Make in-memory computing as easy as np.dot !*

In memory computing technology is a potential solution to overcome the von Neumann bottleneck and to provide significant improvements in energy efficiency. However, for in-memory computing, the algorithm and hardware are highly intertwined, making it challenging to design an optimal circuit. To address this issue, the author developed a Python package named "mPimPy" that embeds hardware simulation into the algorithm. From a user's perspective, mPimPy serves as the equivalent of the Numpy.dot function to implement the matrix multiplication which is the key operation of neural networks, machine learning, signal processing, and scientific computing. By simply replacing the `np.dot` function in your original code with `MapReducedot` and rerunning your program, you can simulate scenarios such as neural networks or any algorithm involving matrix multiplications. Whether you are a computer engineer or a microelectronic scientist, mPimPy enables you to quickly deploy your code on in-memory computing hardware or verify your circuit design solutions. 

## Installing

You can install mPimPy via pip.

```shell
pip install mpimpy
```

## Example

```python
import numpy as np 
from mpimpy import memmat 
from mpimpy import memmatfp 
from mpimpy import memmatdp 
import matplotlib.pyplot as plt 

##*************************Define the Relative Error****************************

def RE(ytest, ypred):
    return np.sqrt(np.sum((ytest-ypred)**2))/np.sqrt(np.sum(ytest**2))

##*************************Generate the Input Matrix**************************** 

a = np.random.randn(32, 32)
b = np.random.randn(32, 32)
c = np.dot(a, b)

##*************************Initialize the Matrix Engine************************* 
'''
The following codes are to initialize the matrix engine, where the parameters are the same as the memristor crossbar array.
'''

dpe_dp = memmatdp.diffpairdpe(HGS=1e-5, LGS=1e-7, g_level=16, var=0.05, vnoise = 0, wire_resistance=2.93,
                            rdac=256, radc=1024, vread=0.1, array_size=(32, 32))

dpe_int = memmat.bitslicedpe(HGS=1/1.3e5, LGS=1/2.1e6, g_level=16, var=0.05, vnoise = 0, wire_resistance=2.93, 
                             rdac=256, radc=1024, vread=0.1, array_size=(32, 32))

dpe_fp = memmatfp.fpmemdpe(HGS=1e-5, LGS=1e-7, g_level=16, var=0.05, vnoise = 0, wire_resistance=2.93,
                            rdac=256, radc=1024, vread=0.1, array_size=(32, 32))

##****************************************************************************** 

##********************Perform the Matrix Multiplication************************* 
'''
The following codes are to perform the matrix multiplication by the software and hardware, respectively. 
The functions are equivalent to the np.dot() function in numpy. Therefore, when you perform the IMC software-hardware co-simulation, what you need to do is to 
replace the "np.dot" in the original program with one of the following four instructions after intializing the matrix engine. So easy, right?
'''
c_df_hardware = dpe_dp.MapReduceDot(a, b)  

c_int_software = dpe_int.BitSliceVMM(a, b, xblk=[1,1,2,4], mblk=[1,1,2,4])
c_int_hardware = dpe_int.MapReduceDot(a, b, xblk=[1 for i in range(8)], mblk=[1 for i in range(8)], wire_factor=False)   #Activate the physical simulation core, wire_factor=True
c_int_hardware_m = dpe_int.MapReduceDot(a, b, xblk=[1,1,2,4], mblk=[1,1,2,4], wire_factor=False)   #Activate the physical simulation core, wire_factor=True

c_fp_software = dpe_fp.fpvmm(a,b, xblk=[1 for i in range(23)], mblk=[1 for i in range(23)], bw_e=8)
c_fp_hardware = dpe_fp.MapReduceDot(a, b, xblk=[1 for i in range(23)], mblk=[1 for i in range(23)], bw_e=8)  

##*******************Plot the results and calculate the RE***********************
plt.subplot(311)
plt.scatter(c.reshape(-1), c_df_hardware.reshape(-1), s=1)
plt.plot(c.reshape(-1), c.reshape(-1), 'r')
plt.title('INT4 diffpair')
plt.xlabel('True')
plt.ylabel('diffpair VMM')

plt.subplot(323)
plt.scatter(c.reshape(-1), c_int_software.reshape(-1), s=1)
plt.plot(c.reshape(-1), c.reshape(-1), 'r')
plt.title('INT8 software')
plt.xlabel('True')
plt.ylabel('Bit-sliced VMM')

plt.subplot(324)
plt.scatter(c.reshape(-1), c_int_hardware.reshape(-1), s=1)
plt.plot(c.reshape(-1), c.reshape(-1), 'r')
plt.title('INT8 hardware')
plt.xlabel('True')
plt.ylabel('Bit-sliced VMM')

plt.subplot(325)
plt.scatter(c.reshape(-1), c_fp_software.reshape(-1), s=1)
plt.plot(c.reshape(-1), c.reshape(-1), 'r')
plt.title('FP32 software')
plt.xlabel('True')
plt.ylabel('Bit-sliced VMM')

plt.subplot(326)
plt.scatter(c.reshape(-1), c_fp_hardware.reshape(-1), s=1)
plt.plot(c.reshape(-1), c.reshape(-1), 'r')
plt.title('FP32 hardware')
plt.xlabel('True')
plt.ylabel('Bit-sliced VMM')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print('PimPy is installed successfully!')
print('The relative error of the four cases are as follows:')
print('INT4 diffpair RE: ', RE(c, c_df_hardware))
print('INT8 software RE: ', RE(c, c_int_software))
print('INT8 hardware RE: ', RE(c, c_int_hardware))
print('INT8 hardware (m) RE: ', RE(c, c_int_hardware_m))
print('FP32 software RE: ', RE(c, c_fp_software))
print('FP32 hardware RE: ', RE(c, c_fp_hardware))
```

## Release Note

### Fixed

Fixed several bugs from the previous version.

### Optimized

Improved the matrix pre-processing algorithm by combing block-wise pre-processing and vector-wise strategies, reducing errors in **quantization** and **pre-alignment**.

## Author

Author: Ling Yang

Email: [3299285328@qq.com](mailto:3299285328@qq.com)

## Contributors

- Houji Zhou, [1499403578@qq.com](mailto:1499403578@qq.com), [zhouhouji (houjizhou) · GitHub](https://github.com/zhouhouji)

- Zhiwei Zhou, [1548384176@qq.com](mailto:1548384176@qq.com)

- Yuyang Fu, [412983100@qq.com](mailto:412983100@qq.com)

## Maintainer

- Maintainer: Researchers from Prof. Xiangshui Miao and Prof. Yi Li's group at HUST

- Affiliation: Huazhong University of Science and Technology, School of Integrated Circuit,  [Institute of Information Storage Materials and Devices (hust.edu.cn)](http://ismd.hust.edu.cn/)

Any advice and criticism are highly appreciated on this package. Naturally, you can also modify the source code to suit your needs. In upcoming versions, we plan to continually incorporate the latest research findings into mPimPy. Stay tuned!

## References

[1]    [Zidan M A, Jeong Y J, Lee J, et al. A general memristor-based partial differential equation solver[J]. Nature Electronics, 2018, 1(7): 411-420.](https://www.nature.com/articles/s41928-018-0100-6)

[2]     [Li C, Hu M, Li Y, et al. Analogue signal and image processing with large
 memristor crossbars[J]. Nature electronics, 2018, 1(1): 52-59.](https://www.nature.com/articles/s41928-017-0002-z)
