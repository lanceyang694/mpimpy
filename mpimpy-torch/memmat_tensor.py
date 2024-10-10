# -*- coding:utf-8 -*-
# @File  : memmat_tensor.py
# @Author: Zhou
# @Date  : 2024/6/27

"""
this is a new version of the memmat_tensor.py
we use the tensor to realize the dot product, and only consider the INT format data
this version is more efficient than the previous version
"""

import torch
from matplotlib import pyplot as plt
from data_formats import SlicedData
from utils import RE, bfp_map_tensor, quant_map_tensor, dot_2d

import time


class DPETensor(object):
    """
    use the bit slice method to realize PDE using tensor
    realize the INT format data
    """
    def __init__(
            self, HGS=1e-4, LGS=1e-8, g_level=1024, var=0.05, vnoise=0.05, wire_resistance=2.93,
            rdac=2**12, radc=2**30, vread=0.1, array_size=(64, 64), input_size=(64, 64)):
        """
        :param HGS: the high conductance state
        :param LGS: the low conductance state
        :param g_level: the number of the conductance levels
        :param var: the variance of the noise
        :param vnoise: the noise of the voltage
        :param wire_resistance: the resistance of the wire
        :param rdac: the resolution of the DAC
        :param radc: the resolution of the ADC
        :param vread: the read voltage
        :param array_size: the size of the array
        :param input_size: the size of the input data
        """

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
        self.input_size = input_size

        if self.radc < 2:
            raise ValueError('The resolution of the ADC should be larger than 1!')
        if self.rdac < 2:
            raise ValueError('The resolution of the DAC should be larger than 1!')
        if self.g_level < 2:
            raise ValueError('The number of the conductance levels should be larger than 1!')
        if self.LGS >= self.HGS:
            raise ValueError('The low conductance state should be smaller than the high conductance state!')

    def __call__(self, x:SlicedData, mat:SlicedData, wire_factor=False):
        return self.MapReduceDot(x, mat, wire_factor)

    def MapReduceDot(self, x:SlicedData, mat:SlicedData, wire_factor=False):
        """
        use the MapReduce method to realize the dot product
        :param x: the input tensor with shape (slice, m, n)
        :param mat: the weight tensor with shape (slice, m, p)
        :param wire_factor: whether consider the wire resistance
        :return: the output tensor with shape (m, p)
        """
        if mat.device.type != x.device.type:
            raise ValueError('The input data and weight data should be in the same device!')
        if x.quantized_data.shape[-1] != mat.quantized_data.shape[-2]:
            raise ValueError('The input data mismatches the shape of weight data!')
        if wire_factor:
            raise NotImplementedError('The wire_factor is not supported in the tensor version!')
        else:
            result = self._dot(x, mat)
        return result

    def _num2R(self, mat:SlicedData):
        # convert the weight data to the resistance and add the noise
        # input dimension (num_divide,num_divide_col_b, slice_b ,n, p)
        # output dimension (num_divide,num_divide_col_b, slice_b ,n, p)
        Q_G = (self.HGS - self.LGS) / (self.g_level - 1)
        max_weights = mat.sliced_max_weights.reshape(1, 1, -1, 1, 1)
        G = torch.round(mat.sliced_data / max_weights * (self.g_level - 1)) * Q_G + self.LGS
        r = torch.exp(torch.normal(0, self.var, G.shape, device=mat.device))
        return G * r

    def _num2V(self, x:SlicedData):
        # convert input data to the voltage (vread)
        xmax = x.sliced_max_weights
        # without batch, the shape is (num_divide_row_a,num_divide, slice_a ,m, n)
        # or (num_divide_row_a,num_divide, slice_a, dbfp_slice, m, n)
        if len(x.shape) == 2:       
            if x.sliced_data.dim() == 5:
                xmax = xmax.reshape(1, 1, -1, 1, 1)
            elif(x.sliced_data.dim() == 6):
                xmax = xmax.reshape(1, 1, -1, 1, 1, 1)
        # with batch, the shape is (batch,num_divide_row_a,num_divide, slice_a ,m, n)
        #                       or (batch,num_divide_row_a,num_divide, slice_a ,dbfp_slice,m, n)
        elif len(x.shape) == 3:     
            if x.sliced_data.dim() == 6:
                xmax = xmax.reshape(1, 1, 1, -1, 1, 1)
            elif x.sliced_data.dim() == 7:
                xmax = xmax.reshape(1, 1, 1, -1, 1, 1, 1)
        else:
            raise ValueError('The input data dimension is not supported!')
        V_in = self.vread * torch.round(x.sliced_data / xmax * (self.rdac - 1)) / (self.rdac - 1)
        return V_in

    def _dot(self, x:SlicedData, mat:SlicedData):
        '''
        calculate the dot product of x and m
        :param x: the input tensor with shape (slice, m, n)
        :param mat: the weight tensor with shape (slice, n, p)
        :return: the output tensor with shape (m, p)
        '''
        G = self._num2R(mat)
        Vin = self._num2V(x)

        if len(x.shape) == 2:
            if x.dbfp_en is True:
                Vin = Vin.permute(2,0,1,3,4,5)
                Vin = Vin[:,:,:,0,:,:]*2.**(x.e_bias[:,:,0,:,:]-x.e_bias[:,:,1,:,:])+Vin[:,:,:,1,:,:]
                Vin = Vin.permute(1,2,0,3,4)
            I = dot_2d(Vin, G - self.LGS)
            adcRef = (self.HGS - self.LGS) * self.vread * Vin.shape[-1]
            I = torch.round(I / adcRef * (self.radc - 1)) / (self.radc - 1)
            QG = (self.HGS - self.LGS) / (self.g_level - 1)
            temp = torch.mul(I, x.sliced_max_weights.reshape(1, 1, 1, -1, 1, 1, 1))
            temp = torch.round(torch.mul(temp, mat.sliced_max_weights.reshape(1, 1, 1, 1, -1, 1, 1))
                                    / QG / self.vread / (self.g_level - 1) * adcRef)
            shift_weights = torch.zeros((len(x),len(mat)), device=x.device)
            
            for i in range(len(x)):
                shift_weights[i] = x.sliced_weights[i] * mat.sliced_weights
            # add the shift weights to the calculated result
            out = torch.mul(temp.reshape(temp.shape[0],temp.shape[1],temp.shape[2], -1, temp.shape[5], temp.shape[6]),
                            shift_weights.reshape(1, 1, 1, -1, 1, 1))
            out = out.sum(dim=3) 
            if x.bw_e is None:
            # The product of max_data for each small piece of the matrix
                out_block_max = torch.einsum("nmij, mpij->nmpij", x.max_data, mat.max_data)
                out = (out * out_block_max
                        / (2 ** (sum(x.slice_method) - 1) - 1) / (2 ** (sum(mat.slice_method) - 1) - 1))
            else:
                if x.dbfp_en is True:
                    out_block_e_bias = torch.einsum("nmij, mpij->nmpij",2.**x.e_bias[:,:,1,:,:], 2.**mat.e_bias)
                else:
                    out_block_e_bias = torch.einsum("nmij, mpij->nmpij",2.**x.e_bias, 2.**mat.e_bias)
                out = out* out_block_e_bias*2.**(4-sum(x.slice_method)-sum(mat.slice_method))
            out = out.sum(dim=1)
            out = out.permute(0, 2, 1, 3)
            out = out.reshape(out.shape[0] * out.shape[1], out.shape[2] * out.shape[3])
            result = out[:x.shape[0],:mat.shape[1]]
            return result
        elif len(x.shape) == 3:     #三维可能会出错， todo：调试
            if x.dbfp_en is True:
                Vin = Vin.permute(3, 0, 1, 2, 4, 5, 6)
                Vin = Vin[:, :, :, :, 0, :, :] * 2. ** (x.e_bias[:, :, :, 0, :, :] -
                                                        x.e_bias[:, :, :, 1, :, :]) + Vin[ :, :, :, :, 1, :, :]
                Vin = Vin.permute(1, 2, 3, 0, 4, 5)
            I = dot_2d(Vin, G - self.LGS)
            adcRef = (self.HGS - self.LGS) * self.vread * Vin.shape[-1]
            I = torch.round(I / adcRef * (self.radc - 1)) / (self.radc - 1)
            QG = (self.HGS - self.LGS) / (self.g_level - 1)
            temp = torch.mul(I, x.sliced_max_weights.reshape(1, 1, 1, 1, -1, 1, 1, 1))
            temp = torch.round(torch.mul(temp, mat.sliced_max_weights.reshape(1, 1, 1, 1, 1, -1, 1, 1))
                                    / QG / self.vread / (self.g_level - 1) * adcRef)
            shift_weights = torch.zeros((len(x),len(mat)), device=x.device)
            
            for i in range(len(x)):
                shift_weights[i] = x.sliced_weights[i] * mat.sliced_weights
            # add the shift weights to the calculated result

            out = torch.mul(temp.reshape(temp.shape[0],temp.shape[1],temp.shape[2],temp.shape[3],
                                         -1, temp.shape[6], temp.shape[7]),
                            shift_weights.reshape(1, 1, 1, 1, -1, 1, 1))
            out = out.sum(dim=4) 
            if x.bw_e is None:
            # The product of max_data for each small piece of the matrix
                out_block_max = torch.einsum("bnmij, mpij->bnmpij", x.max_data, mat.max_data)
                out = (out * out_block_max
                        / (2 ** (sum(x.slice_method) - 1) - 1) / (2 ** (sum(mat.slice_method) - 1) - 1))
            else:
                if x.dbfp_en is True:
                    out_block_e_bias = torch.einsum("bnmij, mpij->bnmpij", 2. ** x.e_bias[:,:,:,1,:,:], 2.**mat.e_bias)
                else:
                    out_block_e_bias = torch.einsum("bnmij, mpij->bnmpij", 2. ** x.e_bias, 2.**mat.e_bias)
                out = out* out_block_e_bias*2.**(4-sum(x.slice_method)-sum(mat.slice_method))
            out = out.sum(dim=2).permute(0, 1, 3, 2, 4)
            out = out.reshape(out.shape[0],out.shape[1] * out.shape[2], out.shape[3] * out.shape[4])
            result = out[:out.shape[0],:x.shape[1],:mat.shape[1]]
            return result

    def slice_data(self, mat, slice_method, bw_e=None, input_en=False, dbfp_en=False):
        """
        slice the data using the slice method
        :param mat: the data to be sliced, 3D tensor, the shape is (batch, row, col)
        :param slice_method: the slice method, tensor or list
        :param bw_e: the width of the exponent, if bw_e is None, then the data is INT format
        :param input_en: if input_en is True, then the input data and weight data has different size
        :param dbfp_en: if dbfp_en is True, then the input data is dbfp format
        :return:
                data_int: the sliced data in INT format, the shape is (batch, divide_num, slice, row, col)
                mat_data: the data quantized by the slice method, the shape is the same as the input data
                max_mat: the max value of the input data for each slice, the shape is (batch, divide_num, 1, 1, 1)
                e_bias: the bias of the exponent, the shape is (batch, divide_num, slice)
        """
        # take all the input as 4D tensor
        unsqueezed = False

        if len(mat.shape) == 2:
            mat = mat.unsqueeze(0)
            unsqueezed = True

        if input_en:
            size = self.input_size
        else:
            size = self.array_size

             # the difference between the transpose and the non-transpose is the slice direction
        if mat.shape[1] % size[0] == 0:
            num_divide_row = int(mat.shape[1] / size[0])
        else:
            num_divide_row = int(mat.shape[1] / size[0]) + 1
        if mat.shape[2] % size[1] == 0:
            num_divide_col = int(mat.shape[2] / size[1])
        else:
            num_divide_col = int(mat.shape[2] / size[1]) + 1

        temp_mat = torch.zeros((mat.shape[0], num_divide_row * size[0],num_divide_col * size[1] ), device=mat.device)
        temp_mat[:, :mat.shape[1], :mat.shape[2]] = mat

        # transpose is used to make temp_mat as (mat.shape[0],num_divide_col,num_divide_row, size[1],  size[0])
        temp_mat = temp_mat.reshape(mat.shape[0], num_divide_row, size[0], num_divide_col, size[1]).transpose(2, 3)
        if bw_e:    # define the FP_map_tensor function
            data_int, mat_data, max_mat, e_bias = bfp_map_tensor(temp_mat, slice_method, bw_e, dbfp_en)
        else:
            data_int, mat_data, max_mat, e_bias = quant_map_tensor(temp_mat, slice_method)
        # the transpose is used to make the data_int is the same as the input data
        mat_data = mat_data.transpose(2,3).reshape(mat.shape[0],
                                                   num_divide_row*size[0],
                                                   num_divide_col*size[1])[:,:mat.shape[1],:mat.shape[2]]
            
        # remove the unsqueezed dimension
        if unsqueezed:
            data_int = data_int.squeeze(0)
            mat_data = mat_data.squeeze(0)
            max_mat = max_mat.squeeze(0)
            if e_bias is not None:
                e_bias = e_bias.squeeze(0)
        
        return data_int, mat_data, max_mat, e_bias , dbfp_en

    def _test(self, x:SlicedData, mat:SlicedData):
        temp = dot_2d(x.sliced_data.float(), mat.sliced_data.float())
        shift_weights = torch.zeros((len(x), len(mat)), device=x.device)
        for i in range(len(x)):
            shift_weights[i] = x.sliced_weights[i] * mat.sliced_weights
        out = torch.mul(temp.reshape(temp.shape[0], -1, temp.shape[3], temp.shape[4]),
                        shift_weights.reshape(1, -1, 1, 1))
        out = (out.sum(dim=1) * x.max_data * mat.max_data
               / (2 ** (sum(x.slice_method) - 1) - 1) / (2 ** (sum(mat.slice_method) - 1) - 1))
        return out.sum(dim=0)

if __name__ == '__main__':
    tb_mode = 1
    if tb_mode == 0:
        x_data = torch.randn(1000, 100)
        mat_data = torch.randn(100, 800)
        mblk = xblk = torch.tensor(10 * [1] + 5 * [2])
        mat = SlicedData(mblk, bw_e=8)
        x = SlicedData(xblk, bw_e=8, input_en=True)
        engine = DPETensor(var=0.0, array_size=(32,32), input_size=(32,32))
        mat.slice_data_imp(engine, mat_data)
        x.slice_data_imp(engine, x_data)
        start = time.time()
        result = engine(x, mat).numpy()
        end = time.time()
        print("Tensor time: ", end - start)

        rel_result = torch.matmul(x_data, mat_data).numpy()
        print(RE(result, rel_result))
        plt.scatter(rel_result.reshape(-1), result.reshape(-1))
        plt.xlabel('Expected Value of Dot Product')
        plt.ylabel('Measured Value of Dot Product')
        plt.show()

    elif tb_mode == 1:
        torch.manual_seed(42)
        device = torch.device('cuda:0')
        x_data = torch.randn(3, 1000, 1000, device=device)
        mat_data = torch.randn(1000, 1200, device=device)
        xblk = torch.tensor([1, 1, 2, 4])
        mblk = torch.tensor([1, 1, 2, 4])
        
        mat = SlicedData(mblk, device=device)
        x = SlicedData(xblk, device=device, input_en=True)

        engine = DPETensor(var=0.05, array_size=(64,64), input_size=(64,64))
        x.slice_data_imp(engine, x_data)
        mat.slice_data_imp(engine, mat_data)
        start = time.time()
        result = engine(x, mat)
        end = time.time()
        print("Tensor time: ", end - start)
        result = result.cpu().numpy()

        rel_result = torch.matmul(x_data, mat_data).cpu().numpy()

        print(RE(result, rel_result))
        plt.scatter(rel_result.reshape(-1), result.reshape(-1))
        plt.xlabel('Expected Value of Dot Product')
        plt.ylabel('Measured Value of Dot Product')
        plt.show()
