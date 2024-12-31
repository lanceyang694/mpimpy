# memmat

"""

@Author: Ling Yang
email: 3299285328@qq.com
Huazhong University of Science and Technology, School of Integrated Circuits
 
"""

import numpy as np 
from mpimpy.crossbar import crossbar


class bitslicedpe:
    
    def __init__(
        self, 
        HGS=1e-5, LGS=1e-7, g_level=16, var=0.1, vnoise = 0.05, wire_resistance=2.93, 
        rdac=256, radc=1024, vread=0.1, array_size=(32, 32)):
        
        self.HGS = HGS 
        self.LGS = LGS 
        self.g_level = g_level 
        self.var = var 
        self.vnoise = vnoise
        self.wire_resistance = wire_resistance 
        self.rdac = rdac 
        self.radc = radc 
        self.vread = vread 
        self.array_size = array_size 
        
        
    def __QuantMap(self, mat, blk=[1, 1, 2, 4]):
        
        assert blk[0]==1
        bits = np.sum(blk)
        
        n_blk = len(blk)
        
        if np.max(np.abs(mat)) == 0:
            data_int = np.zeros((n_blk, mat.shape[0], mat.shape[1]))
            
        else:
            matq = np.round(mat/np.max(np.abs(mat)) * (2**(bits-1)-1))
            matq[np.where(matq<0)] = 2**bits + matq[np.where(matq<0)]
            matq = matq.astype('int')
            data_int = np.zeros((n_blk, mat.shape[0], mat.shape[1]))

            b = 0
            for i in range(n_blk):
                data_int[i,:,:] = ((matq-matq%2**b)%2**(b+blk[-1-i]))>>b
                b+=blk[-1-i]
            
        return data_int
    
    def BitSliceVMM(self, xin, matin, xblk=[1,1,2,4], mblk=[1,1,2,4]):     #注意x必须为二维向量 
    
        if len(xin.shape)==1: 
            x = xin.reshape((1, len(xin)))
        else:
            x = xin*1
            
        if len(matin.shape)==1: 
            mat = matin.reshape((1, len(matin))).T
        else:
            mat = matin*1

        mdata = self.__QuantMap(mat, blk=mblk) 
        xdata = self.__QuantMap(x, blk=xblk) 
        
        mbits = np.sum(mblk) 
        xbits = np.sum(xblk) 
        
        n_mblk = len(mblk) 
        n_xblk = len(xblk) 
        
        out = np.zeros((mat.shape[0], x.shape[1]))
        wi = 0
        
        for i in range(n_xblk):
            out1 = np.zeros((mat.shape[0], x.shape[1]))
            wj = 0
            
            for j in range(n_mblk): 
                if j==(n_mblk-1): 
                    out1 = out1 - np.dot(xdata[i,:, :], mdata[j ,:, :])*2**wj
                else:
                    out1 = out1 + np.dot(xdata[i,:, :], mdata[j ,:, :])*2**wj
                
                wj += mblk[-1-j]
                
            if i==(n_xblk-1):
                out = out - out1*2**wi
                
            else:
                out = out + out1*2**wi
                
            # print(2**wi)  
            wi += xblk[-1-i] 

        mid_result = out*np.max(np.abs(mat))*np.max(np.abs(x))/(2**(mbits-1)-1)/(2**(xbits-1)-1)

        if (len(xin.shape)==1) or (len(mat.shape)==1):
            final_result = mid_result.reshape(-1)
        else:
            final_result = mid_result
            
        return final_result
    
    
    def Num2V(self, xint, xmax):
        
        vout = self.vread * np.round(xint/xmax * (self.rdac-1))/(self.rdac-1)
        
        return vout
    
    def Num2R(self, matint, mmax):
        
        Q_G = (self.HGS-self.LGS)/(self.g_level-1) 
        mat_gq = np.round(matint*(self.g_level-1)/mmax) 
        G = mat_gq * Q_G + self.LGS 
        
        r = np.random.lognormal(0, self.var, size=matint.shape) 
        G = G * r
        
        return G
    
    def __dot_singleblk(self, Vin, G, xmax, mmax, wire_factor=False): 
        
        if wire_factor: 
            
            I = crossbar.hdot(Vin, 1/G, self.wire_resistance) - crossbar.hdot(Vin, 1/self.LGS*np.ones_like(G), self.wire_resistance)
            maxV = self.vread*np.ones(Vin.shape[1]) 
            minR = 1/self.HGS * np.ones(G.shape[0]) 
            maxR = 1/self.LGS * np.ones(G.shape[0]) 

            adcRef = crossbar.hdot(maxV, minR.reshape(len(minR), 1), self.wire_resistance) - crossbar.hdot(maxV, maxR.reshape(len(maxR), 1), self.wire_resistance)

        else:

            I = np.dot(Vin, G - self.LGS)     

            
            adcRef = (self.HGS - self.LGS) * self.vread * Vin.shape[1]   
        
        Iq = np.round(I/adcRef * (self.radc-1)) / (self.radc-1) 
        QG = (self.HGS - self.LGS) / (self.g_level-1) 
        
        Num = np.round(Iq / QG / self.vread / (self.g_level-1) * xmax * mmax * adcRef)   
        
        return Num 
    
    def __dot(self, x, mat, xblk=[1,1,2,4], mblk=[1,1,2,4], wire_factor=False):
        
        xint = self.__QuantMap(x, blk=xblk) 
        matint = self.__QuantMap(mat, blk=mblk) 
        
        xbits = np.sum(xblk) 
        mbits = np.sum(mblk) 
        
        n_mblk = len(mblk)
        n_xblk = len(xblk)
    
        out = np.zeros((x.shape[0], mat.shape[1])) 
        wi = 0 
        
        for i in range(n_mblk): 
            
            G = self.Num2R(matint[i,:, :], 2**mblk[-1-i]-1)
            out1 = np.zeros((x.shape[0], mat.shape[1])) 
            wj = 0 
            
            for j in range(n_xblk): 
                
                Vin = self.Num2V(xint[j ,:, :], 2**xblk[-1-j]-1)
                
                if j==(n_xblk-1): 
                    out1 = out1 - self.__dot_singleblk(Vin, G, 2**xblk[-1-j]-1, 2**mblk[-1-i]-1, wire_factor) *2**wj
                else: 
                    out1 = out1 + self.__dot_singleblk(Vin, G, 2**xblk[-1-j]-1, 2**mblk[-1-i]-1, wire_factor) *2**wj
                
                wj += xblk[-1-j] 
                
            if i==(n_mblk-1):
                out = out - out1*2**wi
            else:
                out = out + out1*2**wi
                
            wi += mblk[-1-i]

        return out*np.max(np.abs(x))*np.max(np.abs(mat))/(2**(xbits-1)-1)/(2**(mbits-1)-1)
    
        
    def MapReduceDot(self, xin, matin, xblk=[1,1,2,4], mblk=[1,1,2,4], wire_factor=False):
        
        if len(xin.shape)==1: 
            x = xin.reshape((1, len(xin)))
        else:
            x = xin*1
            
        if len(matin.shape)==1: 
            mat = matin.reshape((1, len(matin))).T
        else:
            mat = matin*1
            
        n_row = x.shape[0]
        n_col = mat.shape[1]
        
        polish0 = mat.shape[0] % self.array_size[0] 
        polish1 = mat.shape[1] % self.array_size[1] 
        
        if polish1 != 0: 
            mat = np.hstack((mat, np.zeros((mat.shape[0], self.array_size[1] - polish1))))
             
        if polish0 != 0:
            mat = np.vstack((mat, np.zeros((self.array_size[0] - polish0, mat.shape[1]))))
            x = np.hstack((x, np.zeros((x.shape[0], self.array_size[0] - polish0))))
        
        result = np.zeros((x.shape[0], mat.shape[1])) 
        
        for i in range(int(mat.shape[1]/self.array_size[1])): 
            
            block_out_row = 0 
            
            for j in range(int(mat.shape[0]/self.array_size[0])): 

                operand_x = x[:, j*self.array_size[0] : (j+1)*self.array_size[0]]
                operand_m = mat[j*self.array_size[0] : (j+1)*self.array_size[0], i*self.array_size[1] : (i+1)*self.array_size[1]] 
                block_out_row += self.__dot(operand_x, operand_m, xblk, mblk, wire_factor) 
                
            result[:, i*self.array_size[1] : (i+1)*self.array_size[1]] = block_out_row 
            
        mid_result = result[:n_row, :n_col]
        
        if (len(xin.shape)==1) or (len(mat.shape)==1):
            final_result = mid_result.reshape(-1)
        else:
            final_result = mid_result
            
        return final_result  