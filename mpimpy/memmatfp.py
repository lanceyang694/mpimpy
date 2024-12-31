# memmatfp

"""

@Author: Ling Yang
email: 3299285328@qq.com
Huazhong University of Science and Technology, School of Integrated Circuits 

""" 

import numpy as np 
from mpimpy.crossbar import crossbar

class fpmemdpe:
    
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
        
        assert self.rdac>=2
        assert self.radc>=2
        
        
    def __Dec2FpMap(self, decmat, blk=[1,2,2,2,4,4,4,4], bw_e=8): 
        
        newblk = [1,1] + blk 
        num_blk = len(newblk) 
        
        max_a = np.max(np.abs(decmat))
        e_bia = 0
        
        if max_a>=2:
            while (max_a>=2):
                max_a /= 2
                e_bia += 1
        elif (max_a<1) and (max_a>0):
            while ((max_a<1) and (max_a>0)):
                max_a *= 2
                e_bia -= 1
        else:
            e_bia = 0
            
        decmat_aliE = decmat / 2**e_bia 
        
        decmat_aliE[np.where(decmat_aliE<0)] = 4 + decmat_aliE[np.where(decmat_aliE<0)] 
        
        b = np.zeros((decmat.shape[0], decmat.shape[1], num_blk)) 
        w = 0 
        for i in range(num_blk): 
            w = w + newblk[i] 
            b[:,:,i] = (decmat_aliE / 2**(2-w)).astype('int') 
            decmat_aliE -= b[:,:,i]*(2**(2-w)) 
            
        e_max_range = 2**(bw_e-1) - 1
        
        return np.clip(np.array([e_bia]), -e_max_range, e_max_range), b 
        
    def fpvmm(self, xin, matin, xblk=[1,2,2,2,4,4,4,4], mblk=[1,2,2,2,4,4,4,4], bw_e=8): 
        
        if len(xin.shape)==1: 
            x = xin.reshape((1, len(xin)))
        else:
            x = xin*1
            
        if len(matin.shape)==1: 
            mat = matin.reshape((1, len(matin))).T
        else:
            mat = matin*1
        

        Ex, fpx = self.__Dec2FpMap(x, blk=xblk, bw_e=bw_e) 
        Em, fpm = self.__Dec2FpMap(mat, blk=mblk, bw_e=bw_e)
        
        nxblk = [1, 1]+xblk
        nmblk = [1, 1]+mblk
        
        out = np.zeros((x.shape[0], mat.shape[1]))
        wi = 0
        for i in range(len(nxblk)):
            
            wi += nxblk[i]
            out1 = np.zeros((x.shape[0], mat.shape[1])) 
            wj = 0 
            for j in range(len(nmblk)): 
                wj += nmblk[j] 
                if j==0: 
                    out1 = out1 - np.dot(fpx[:,:,i], fpm[:, :, j])*(2**(2-wj)) 
                else: 
                    out1 = out1 + np.dot(fpx[:,:,i], fpm[:, :, j])*(2**(2-wj))  
                    
            if i==0: 
                out = out - out1*(2**(2-wi)) 
            else: 
                out = out + out1*(2**(2-wi)) 
                
        mid_result = out*(2.**(Ex[0]+Em[0]))
                   
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
            maxV = self.vread * np.ones(Vin.shape[1]) 
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
    
    def __dot(self, x, mat, xblk=[1,2,2,2,4,4,4,4], mblk=[1,2,2,2,4,4,4,4], bw_e=8, wire_factor=False):
        
        Ea, xint = self.__Dec2FpMap(x, blk=xblk, bw_e=bw_e) 
        Eb, matint = self.__Dec2FpMap(mat, blk=xblk, bw_e=bw_e) 
        
        nxblk = [1, 1]+xblk
        nmblk = [1, 1]+mblk
        
        num_xblk = len(nxblk) 
        num_mblk = len(nmblk) 
    
        out = np.zeros((x.shape[0], mat.shape[1])) 
        wi = 0 
        for i in range(num_mblk): 
            
            wi += nmblk[i]
            G = self.Num2R(matint[:,:, i], 2**nmblk[i]-1) 
            out1 = np.zeros((x.shape[0], mat.shape[1])) 
            wj = 0 
            
            for j in range(num_xblk): 
                
                wj += nxblk[j] 
                Vin = self.Num2V(xint[: ,:, j], 2**nxblk[j]-1) 
                
                if j==0: 
                    out1 = out1 - self.__dot_singleblk(Vin, G, 2**nxblk[j]-1, 2**nmblk[i]-1, wire_factor) * 2**(2-wj)
                else: 
                    out1 = out1 + self.__dot_singleblk(Vin, G, 2**nxblk[j]-1, 2**nmblk[i]-1, wire_factor) * 2**(2-wj)
                
            if i==0: 
                out = out - out1* 2**(2-wi) 
            else: 
                out = out + out1* 2**(2-wi) 

        return out * (2.**(Ea[0]+Eb[0])) 
    
    def MapReduceDot(self, xin, matin, xblk=[1,2,2,2,4,4,4,4], mblk=[1,2,2,2,4,4,4,4], bw_e=8, wire_factor=False): 
        
        if len(xin.shape)==1: 
            x = xin.reshape((1, len(xin))) 
        else:
            x = xin
            
        if len(matin.shape)==1: 
            mat = matin.reshape((1, len(matin))).T 
        else:
            mat = matin 
            
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
                block_out_row += self.__dot(operand_x, operand_m, xblk, mblk, bw_e, wire_factor) 
                
            result[:, i*self.array_size[1] : (i+1)*self.array_size[1]] = block_out_row 
            
        mid_result = result[:n_row, :n_col]
        
        if (len(xin.shape)==1) or (len(mat.shape)==1):
            final_result = mid_result.reshape(-1)
        else:
            final_result = mid_result
            
        return final_result
    