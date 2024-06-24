'''
@author: Ling Yang
A simulation demonstration of PIM for Discrete Cosine Transform
'''

from mpimpy import memmat
from mpimpy import memmatfp
from mpimpy import memmatdp
import numpy as np
import matplotlib.pyplot as plt

## Define the hardware parameters

dpe_int = memmat.bitslicedpe(HGS=1/1.3e5, LGS=1/2.23e6, g_level=16, var=0.05, vnoise = 0, wire_resistance=2.93, 
                             rdac=256, radc=256, vread=0.1, array_size=(32, 32))

dpe_dp = memmatdp.diffpairdpe(HGS=1/1.3e5, LGS=1/2.23e6, g_level=16, var=0.05, vnoise = 0, wire_resistance=2.93,
                            rdac=256, radc=1024, vread=0.1, array_size=(32, 32))

dpe_fp = memmatfp.fpmemdpe(HGS=1/1.3e5, LGS=1/2.23e6, g_level=16, var=0.05, vnoise = 0, wire_resistance=2.93,
                            rdac=256, radc=1024, vread=0.1, array_size=(32, 32))

## Define the DCT matrix
def cosine_m(sca):
    A = np.zeros([sca,sca])
    for i in range(sca):
        for j in range(sca):
            if(i == 0):
                x = np.sqrt(1/sca)
            else:
                x = np.sqrt(2/sca)
            A[i][j] = x*np.cos(np.pi*(j+0.5)*i/sca)
            
    return A


dct_m = cosine_m(128)
x = np.linspace(0, 10, 128)
y = np.sin(2*np.pi*x)

## Implement DCT by dot product

z = np.dot(dct_m, y)
z_int4 = dpe_dp.MapReduceDot(y, dct_m.T).T
z_int8 = (dpe_int.MapReduceDot(y, dct_m.T, xblk=[1 for i in range(8)], mblk=[1 for i in range(8)])).T
z_fp16 = (dpe_fp.MapReduceDot(y, dct_m.T, xblk=[1 for i in range(10)], mblk=[1 for i in range(10)])).T

plt.plot(z)
plt.plot(z_int4)
plt.plot(z_int8)
plt.plot(z_fp16)
plt.legend(['ideal', 'INT4', 'INT8', 'FP16'])
# plt.savefig(r'Figures\demo_dct.png')
plt.show()