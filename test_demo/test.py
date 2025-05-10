'''
Version 1.0
@Author: Ling Yang
email: 3299285328@qq.com
03/25, 2025
Huazhong University of Science and Technology, School of Integrated Circuits 
A demo of the use for mpimpy

'''

'''
    _______________________Matrix Engine parameters_____________________________
    memmat: the package for the integer matrix engine
    memmatfp: the package for the floating-point matrix engine
    HGS: high conductance state 
    LGS: low conductance state 
    g_level: number of conductance levels 
    var: device-to-device variability of the conductance 
    vnoise: noise of the input voltage 
    wire_resistance: resistance of the wire 
    rdac: resolution of the DAC 
    radc: resolution of the ADC 
    vread: read voltage 
    array_size: size of the single memristor crossbar array 
    _____________________________________________________________________________
    
    _______________________Matrix Engine functions_______________________________
    xblk: bit slice rule of the input matrix, where the first element denotes the sign bit which must be 1.
    mblk: bit slice rule of the weight matrix, where the first element denotes the sign bit which must be 1.
    For floating-point numbers, the signed bit and integer part are setted to 1 compulsorily.
    bw_e: bit width of the exponent for floating-point numbers, default is 8 (the bit width of the exponent in FP32).
    BitSliceVMM: bit-sliced vector-matrix multiplication performed by the software.
    MapReduceDot: bit-sliced vector-matrix multiplication performed by the hardware.
    _____________________________________________________________________________
    
    _______________________How to use pimpy?_____________________________________
    1. import the numpy package and teh pimpy package
    2. initialize the matrix engine, 
    e.g., integer: dpe_int = memmat.bitslicedpe(HGS=1/1.3e5, LGS=1/2.1e6, g_level=16, var=0.27, vnoise = 0, wire_resistance=2.93, 
                                                rdac=256, radc=256, vread=0.1, array_size=(32, 32))
    e.g., floating-point: dpe_fp = memmatfp.fpmemdpe(HGS=1e-5, LGS=1e-7, g_level=16, var=0.05, vnoise = 0, wire_resistance=2.93, 
                                                rdac=256, radc=1024, vread=0.1, array_size=(32, 32))
    3. perform the matrix multiplication, 
    e.g., c_int_software = dpe_int.MapReduceDot(a, b, xblk=[1,1,1,1,2,4], mblk=[1,1,2,4]),
    meaning that the input matrix a is sliced as [1,1,1,1,2,4] and the weight matrix b is sliced as [1,1,2,4]
    
    Pimpy, makes the in-memory computing as easy as numpy.dot()
    
    OK, let's start the happy journey in the world of In-memory computing!
    _____________________________________________________________________________

'''

import numpy as np 
from mpimpy import memmat 
from mpimpy import memmatfp 
from mpimpy import memmatdp 
import matplotlib.pyplot as plt 

##*************************Define the Relative Error****************************

def RE(ytest, ypred):
    return np.sqrt(np.sum((ytest-ypred)**2))/np.sqrt(np.sum(ytest**2))

##*************************Generate the Input Matrix**************************** 
np.random.seed(42)
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
c_df_hardware = dpe_dp.MapReduceDot(a, b, wire_factor=False)  

c_int_software = dpe_int.BitSliceVMM(a, b, xblk=[1,1,2,4], mblk=[1,1,2,4])
c_int_hardware = dpe_int.MapReduceDot(a, b, xblk=[1 for i in range(8)], mblk=[1 for i in range(8)], wire_factor=False)   #Activate the physical simulation core, wire_factor=True
c_int_hardware_m = dpe_int.MapReduceDot(a, b, xblk=[1,1,2,4], mblk=[1,1,2,4], wire_factor=False)   #Activate the physical simulation core, wire_factor=True

c_fp_software = dpe_fp.fpvmm(a,b, xblk=[1 for i in range(23)], mblk=[1 for i in range(23)], bw_e=8)
c_fp_hardware = dpe_fp.MapReduceDot(a, b, xblk=[1 for i in range(23)], mblk=[1 for i in range(23)], bw_e=8, wire_factor=False)  

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