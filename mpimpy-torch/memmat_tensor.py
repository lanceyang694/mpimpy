# -*- coding:utf-8 -*-
# @File  : memmat_tensor.py
# @Author: Zhou
# @Date  : 2024/6/27

'''
this is a new version of the memmat_tensor.py
we use the tensor to realize the dot product, and only consider the INT format data
this version is more efficient than the previous version
'''

import torch
import numpy as np
from data_formats import SlicedData

def ABSE(ytest, ypred):
    return np.sum(np.abs((ytest-ypred)/ytest))/(ytest.shape[0] * ytest.shape[1])

def quant_map_tensor(mat, blk=(1, 1, 2, 4)):
    # convert the original data to the quantized data
    # input is a three-dimensional tensor (divide_num, row, col)
    quant_data_type = torch.int8 if max(blk)<=8 else torch.int16
    if len(mat.shape) == 3:
        assert blk[0] == 1
        bits = sum(blk)

        max_mat = torch.max(torch.max(torch.abs(mat), dim=1, keepdim=True)[0], dim=2, keepdim=True)[0] #(divide_num, 1, 1)
        matq = torch.round(mat / max_mat * (2 ** (bits - 1) - 1)).int()   # (divide_num, row, col), int data
        # record quantized data
        mat_data = matq / (2 ** (bits - 1) - 1) * max_mat

        # use location to reduce the function where
        location = torch.where(matq < 0)
        matq[location] = 2 ** bits + matq[location]
        data_int = torch.empty((mat.shape[0], len(blk), mat.shape[1], mat.shape[2]), device=mat.device, dtype=quant_data_type)
        b = 0
        for idx, bits in enumerate(blk):
            data_int[:, idx, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - idx])) >> b
            b += blk[-1 - idx]
    elif len(mat.shape) == 2:
        assert blk[0] == 1
        bits = sum(blk)
        max_mat = None
        if torch.max(torch.abs(mat)) == 0:
            data_int = torch.zeros((len(blk), mat.shape[0], mat.shape[1]), device=mat.device, dtype=quant_data_type)
            mat_data = torch.zeros((mat.shape[0], mat.shape[1]), device=mat.device)
        else:
            matq = torch.round(mat / torch.max(torch.abs(mat)) * (2 ** (bits - 1) - 1)).int()

            # use location to reduce the function where
            location = torch.where(matq < 0)
            matq[location] = 2 ** bits + matq[location]
            # record quantized data
            mat_data = matq / (2 ** (bits - 1) - 1) * torch.max(torch.abs(mat))
            data_int = torch.empty((len(blk), mat.shape[0], mat.shape[1]), device=mat.device, dtype=quant_data_type)
            b = 0
            for idx, bits in enumerate(blk):
                data_int[idx, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - idx])) >> b
                b += blk[-1 - idx]
    else:
        raise NotImplementedError
    return data_int, mat_data, max_mat

import time

def dot_2d(a, b):
    """
    use einsum to calculate the cross 2D product
    :param a: tensor with shape (divide_num, slice_a, m, n)
    :param b: tensor with shape (divide_num, slice_b, n, p)
    """
    return torch.einsum("mijk, mskl->misjl", a, b)


class DPETensor(object):
    '''
    use the bit slice method to realize PDE using tensor
    realize the INT format data
    '''
    def __init__(
            self, HGS=1e-5, LGS=1e-7, g_level=16, var=0.05, vnoise=0.05, wire_resistance=2.93,
            rdac=256, radc=1024, vread=0.1, array_size=(64, 64)):

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
        '''
        use the MapReduce method to realize the dot product
        :param x: the input tensor with shape (slice, m, n)
        :param x_slice_method: the slice method of the input tensor
        :param mat: the weight tensor with shape (slice, m, p)
        :param wire_factor: whether consider the wire resistance
        :return: the output tensor with shape (m, p)
        '''

        if mat.device.type != x.device.type:
            raise ValueError('The input data and weight data should be in the same device!')
        if x.quantized_data.shape[1] != mat.quantized_data.shape[0]:
            raise ValueError('The input data mismatches the shape of weight data!')
        if wire_factor:
            raise NotImplementedError('The wire_factor is not supported in the tensor version!')
        else:
            result = self._dot(x, mat)
        return result

    def _num2R(self, mat:SlicedData):
        # convert the weight data to the resistance and add the noise
        # input dimension (divide_num, slice, row, col)
        # output dimension (divide_num, slice, row, col)
        Q_G = (self.HGS - self.LGS) / (self.g_level - 1)
        max_weights = mat.sliced_max_weights.reshape(1, -1, 1, 1)
        G = torch.round(mat.sliced_data / max_weights * (self.g_level - 1)) * Q_G + self.LGS
        r = torch.exp(torch.normal(0, self.var, G.shape, device=mat.device))
        return G * r

    def _num2V(self, x:SlicedData):
        # convert input data to the voltage (vread)
        # if x dim is 4, the dimension means (divide_num, slice, row, col)
        # and the output dimension is (divide_num, slice, row, col)
        # if x dim is 5, the dimension means (batch, divide_num, slice, row, col)
        # and the output dimension is (batch, divide_num, slice, row, col)
        xmax = x.sliced_max_weights
        if len(x.shape) == 2:
            xmax = xmax.reshape(1, -1, 1, 1)
        elif len(x.shape) == 3:
            xmax = xmax.reshape(1, -1, 1, 1, 1)
        else:
            raise ValueError('The input data dimension is not supported!')
        V_in = self.vread * torch.round(x.sliced_data / xmax * (self.rdac - 1)) / (self.rdac - 1)
        return V_in

    def _dot(self, x:SlicedData, mat:SlicedData):
        '''
        calculate the dot product of x and m
        :param x: the input tensor with shape (slice, m, n)
        :param m: the weight tensor with shape (slice, n, p)
        :return: the output tensor with shape (m, p)
        '''
        G = self._num2R(mat)
        Vin = self._num2V(x)

        I = dot_2d(Vin, G - self.LGS)
        adcRef = (self.HGS - self.LGS) * self.vread * Vin.shape[-1]

        I = torch.round(I / adcRef * (self.radc - 1)) / (self.radc - 1)
        QG = (self.HGS - self.LGS) / (self.g_level - 1)

        # INT运算本身均为整数，因此结果也是整数，对每一次的结果取整，能大幅提升计算的精度
        # operate input slice
        temp = torch.mul(I, x.sliced_max_weights.reshape(1, -1, 1, 1, 1))
        temp = torch.round(torch.mul(temp, mat.sliced_max_weights.reshape(1, 1, -1, 1, 1))
                               / QG / self.vread / (self.g_level - 1) * adcRef)

        # obtain the shift weights
        shift_weights = torch.zeros((len(x),len(mat)), device=x.device)
        for i in range(len(x)):
            shift_weights[i] = x.sliced_weights[i] * mat.sliced_weights
        # add the shift weights to the calculated result
        out = torch.mul(temp.reshape(temp.shape[0], -1, temp.shape[3], temp.shape[4]),
                        shift_weights.reshape(1,-1, 1, 1))
        out = (out.sum(dim=1) * x.max_data * mat.max_data
                / (2 ** (sum(x.slice_method) - 1) - 1) / (2 ** (sum(mat.slice_method) - 1) - 1))
        return out.sum(dim=0)

    def slice_data(self, mat, slice_method, transpose=False, bw_e=None):
        # use the slice method to slice the data
        # if transpose is True, then the slice is on the column direction
        # if transpose is False, then the slice is on the row direction
        if transpose:
            # input data is divided by the array size in the weight
            if mat.shape[1] % self.array_size[0] == 0:
                end_flag = int(mat.shape[1] / self.array_size[0])
            else:
                end_flag = int(mat.shape[1] / self.array_size[0]) + 1
            # result = torch.zeros((len(slice_method), mat.shape[0], mat.shape[1]), device=mat.device)
            temp_mat = torch.zeros((mat.shape[0], end_flag * self.array_size[0]), device=mat.device)
            temp_mat[:, :mat.shape[1]] = mat
            temp_mat = temp_mat.reshape(mat.shape[0], end_flag, self.array_size[0]).transpose(0,1)
            if bw_e:
                data_int, mat_data, max_mat = quant_map_tensor(temp_mat, slice_method)
            else:
                data_int, mat_data, max_mat = quant_map_tensor(temp_mat, slice_method)
            mat_data = mat_data.transpose(0,1).reshape(mat.shape[0], -1)[:, :mat.shape[1]]
        else:
            if mat.shape[0] % self.array_size[0] == 0:
                end_flag = int(mat.shape[0] / self.array_size[0])
            else:
                end_flag = int(mat.shape[0] / self.array_size[0]) + 1

            # todo: optimize the following code using inplace operation
            temp_mat = torch.empty((end_flag * self.array_size[0], mat.shape[1]), device=mat.device)
            temp_mat[:mat.shape[0]] = mat
            temp_mat[mat.shape[0]:] = 0

            temp_mat = temp_mat.reshape(end_flag, self.array_size[0], mat.shape[1])
            data_int, mat_data, max_mat = quant_map_tensor(temp_mat, slice_method)
            mat_data = mat_data.reshape(-1, mat.shape[1])[:mat.shape[0]]
        # mat_data.requires_grad = True
        return data_int, mat_data, max_mat

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
    from matplotlib import pyplot as plt
    
    tb_mode = 1
    if tb_mode == 0:
        torch.manual_seed(42)
        x_data = torch.randn(1000, 1000)
        mat_data = torch.randn(1000, 1200)
        xblk = torch.tensor([1, 1, 4, 4])
        mblk = torch.tensor([1, 1, 4, 4])
        mat = SlicedData(mblk)
        x = SlicedData(xblk)

        engine = DPETensor(var=0.05)
        mat.slice_data_imp(engine, mat_data)
        x.slice_data_imp(engine, x_data, transpose=True)
        start = time.time()
        result = engine(x, mat).numpy()
        end = time.time()
        print("Tensor time: ", end-start)

        rel_result = torch.matmul(x_data, mat_data).numpy()
        plt.scatter(rel_result.reshape(-1), result.reshape(-1))
        plt.xlabel('Expected Value of Dot Product')
        plt.ylabel('Measured Value of Dot Product')
        plt.show()

    elif tb_mode == 1:
        torch.manual_seed(42)
        device = torch.device('cuda:0')
        x_data = torch.randn(1000, 1000, device=device)
        mat_data = torch.randn(1000, 1200, device=device)
        xblk = torch.tensor([1, 1, 4, 4])
        mblk = torch.tensor([1, 1, 4, 4])
        mat = SlicedData(mblk, device=device)
        x = SlicedData(xblk, device=device)

        engine = DPETensor(var=0.0)
        x.slice_data_imp(engine, x_data, transpose=True)
        mat.slice_data_imp(engine, mat_data)
        start = time.time()
        result = engine(x, mat)
        end = time.time()
        print("Tensor time: ", end-start)
        result = result.cpu().numpy()

        rel_result = torch.matmul(x_data, mat_data).cpu().numpy()

        print(ABSE(result, rel_result))
        plt.scatter(rel_result.reshape(-1), result.reshape(-1))
        plt.xlabel('Expected Value of Dot Product')
        plt.ylabel('Measured Value of Dot Product')
        plt.show()
