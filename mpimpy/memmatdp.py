# differential pair
'''
@Author: Ling Yang
email: 3299285328@qq.com
Huazhong University of Science and Technology, School of Integrated Circuits 
Date: 2025/02/01
'''

import numpy as np
from mpimpy.crossbar import crossbar

class diffpairdpe:
    
    def __init__(
        self,
        HGS=1e-5, LGS=1e-7, g_level = 16,
        var=0.02, vnoise=0.05, wire_resistance=2.93, 
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
        
    
    def Num2V(self, data): 
        
        vin = np.zeros(data.shape)
        max_mat = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            max_mat[i] = np.max(np.abs(data[i, :]))
            if max_mat[i] == 0:
                vin[i, :] = np.zeros(data.shape[1])
            else:
                vin[i, :] = self.vread * np.round(data[i, :]/max_mat[i]*(self.rdac - 1)) / (self.rdac - 1)

        
        return (max_mat, np.hstack((vin, -vin)))
    
    def Num2R(self, data): 
        
        q_data = np.zeros(data.shape)
        max_mat = np.zeros(data.shape[1])
        for i in range(data.shape[1]):
            max_mat[i] = np.max(np.abs(data[:, i]))
            if max_mat[i] == 0:
                q_data[:, i] = np.zeros(data.shape[0])
            else:
                q_data[:, i] = np.round(data[:, i]/max_mat[i]*(self.g_level - 1))
            
        up_0 = np.where(q_data>=0)
        low_0 = np.where(q_data<0)
        gp = np.zeros(q_data.shape)
        gn = np.zeros(q_data.shape)
        
        g_unit = (self.HGS - self.LGS)/(self.g_level - 1)
        
        gp[up_0] = q_data[up_0] * g_unit + self.LGS
        gp[low_0] = self.LGS
        gn[up_0] = self.LGS
        gn[low_0] = np.abs(q_data[low_0]) * g_unit + self.LGS
    
        r1 = np.random.lognormal(0, self.var, size=data.shape)
        r2 = np.random.lognormal(0, self.var, size=data.shape)
    
        gp = gp*r1
        gn = gn*r2
        
        return (max_mat, np.vstack((gp, gn))) 
    
    def __dot(self, x, mat, wire_factor=False): 
          
        max_x, vx = self.Num2V(x) 
        max_m, gmat = self.Num2R(mat) 
        g_unit = (self.HGS - self.LGS) / (self.g_level - 1) 
        
        if wire_factor:
            I = crossbar.hdot(vx, 1/gmat, self.wire_resistance)
            # Iref = (self.HGS - self.LGS) * self.vread * x.shape[1]

            maxV = np.concatenate((self.vread*np.ones(x.shape[1]), -self.vread*np.ones(x.shape[1])))
            maxR = np.concatenate((1/self.HGS * np.ones(mat.shape[0]), 1/self.LGS * np.ones(mat.shape[0])))
            Iref = crossbar.hdot(maxV, maxR.reshape(len(maxR), 1), self.wire_resistance)
        else:
            I = np.dot(vx, gmat) 
            Iref = (self.HGS - self.LGS) * self.vread * x.shape[1]

        Iq = np.round(I/Iref * (self.radc-1))/(self.radc-1)
        Num = np.dot(np.diag(max_x), np.dot(Iq, np.diag(max_m))) / g_unit / self.vread / (self.g_level - 1) * Iref

        return Num
    
    def MapReduceDot(self, xin, matin, wire_factor=False):
        
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
        
        if polish0 != 0: 
            mat = np.hstack((mat, np.zeros((mat.shape[0], self.array_size[1] - polish1)))) 
            
        if polish1 != 0:
            mat = np.vstack((mat, np.zeros((self.array_size[0] - polish0, mat.shape[1])))) 
            x = np.hstack((x, np.zeros((x.shape[0], self.array_size[0] - polish0)))) 
        
        result = np.zeros((x.shape[0], mat.shape[1]))
        for i in range(int(mat.shape[1]/self.array_size[1])):
            block_out_row = 0
            for j in range(int(mat.shape[0]/self.array_size[0])):
                operand_x = x[:, j*self.array_size[1] : (j+1)*self.array_size[1]] 
                operand_m = mat[j*self.array_size[0] : (j+1)*self.array_size[0], i*self.array_size[1] : (i+1)*self.array_size[1]] 
                block_out_row += self.__dot(operand_x, operand_m, wire_factor) 
                
            result[:, i*self.array_size[1] : (i+1)*self.array_size[1]] = block_out_row
            
        mid_result = result[:n_row, :n_col]
        
        if (len(xin.shape)==1) or (len(mat.shape)==1):
            final_result = mid_result.reshape(-1)
        else:
            final_result = mid_result
            
        return final_result 
