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
from data_formats import SlicedData
from utils import quant_map_tensor, bfp_map_tensor, RE, dot_2d
import time


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
        if x.quantized_data.shape[-1] != mat.quantized_data.shape[-2]:
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
            xmax = xmax.reshape(1, 1, -1, 1, 1)
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

        if len(x.shape) == 2:
            temp = torch.mul(I, x.sliced_max_weights.reshape(1, -1, 1, 1, 1))
            temp = torch.round(torch.mul(temp, mat.sliced_max_weights.reshape(1, 1, -1, 1, 1))
                               / QG / self.vread / (self.g_level - 1) * adcRef)

            # obtain the shift weights
            shift_weights = torch.zeros((len(x), len(mat)), device=x.device)
            for i in range(len(x)):
                shift_weights[i] = x.sliced_weights[i] * mat.sliced_weights
            # add the shift weights to the calculated result
            out = torch.mul(temp.reshape(temp.shape[0], -1, temp.shape[3], temp.shape[4]),
                            shift_weights.reshape(1, -1, 1, 1))
            if x.bw_e is None:
                out = (out.sum(dim=1) * x.max_data * mat.max_data
                       / (2 ** (sum(x.slice_method) - 1) - 1) / (2 ** (sum(mat.slice_method) - 1) - 1))
            else:
                out = (out.sum(dim=1) * 2. ** (x.e_bias + mat.e_bias + 4 - sum(x.slice_method) - sum(mat.slice_method)))
            return out.sum(dim=0)
        elif len(x.shape) == 3:
            temp = torch.mul(I, x.sliced_max_weights.reshape(1, 1, -1, 1, 1, 1))
            temp = torch.round(torch.mul(temp, mat.sliced_max_weights.reshape(1, 1, 1, -1, 1, 1))
                               / QG / self.vread / (self.g_level - 1) * adcRef)

            # obtain the shift weights
            shift_weights = torch.zeros((len(x), len(mat)), device=x.device)
            for i in range(len(x)):
                shift_weights[i] = x.sliced_weights[i] * mat.sliced_weights
            # add the shift weights to the calculated result
            out = torch.mul(temp.reshape(temp.shape[0], temp.shape[1], -1, temp.shape[4], temp.shape[5]),
                            shift_weights.reshape(1, 1, -1, 1, 1))
            if x.bw_e is None:
                out = (out.sum(dim=2) * x.max_data * mat.max_data
                       / (2 ** (sum(x.slice_method) - 1) - 1) / (2 ** (sum(mat.slice_method) - 1) - 1))
            else:
                out = (out.sum(dim=2) * 2. ** (x.e_bias + mat.e_bias + 4 - sum(x.slice_method) - sum(mat.slice_method)))
            return out.sum(dim=1)

    def slice_data(self, mat, slice_method, transpose=False, bw_e=None):
        """
        slice the data using the slice method
        :param mat: the data to be sliced, 3D tensor, the shape is (batch, row, col)
        :param slice_method: the slice method, tensor or list
        :param transpose: if transpose is True, then the slice is on the column direction
                            if transpose is False, then the slice is on the row direction
        :param bw_e: the width of the exponent, if bw_e is None, then the data is INT format
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
        if transpose:
            # the difference between the transpose and the non-transpose is the slice direction
            if mat.shape[2] % self.array_size[0] == 0:
                end_flag = int(mat.shape[2] / self.array_size[0])
            else:
                end_flag = int(mat.shape[2] / self.array_size[0]) + 1

            temp_mat = torch.zeros((mat.shape[0], mat.shape[1], end_flag * self.array_size[0]), device=mat.device)
            temp_mat[:, :, :mat.shape[2]] = mat
            # transpose is used to make the end_flag is before the mat.shape[1], because the reshape function can only
            # divide the dimensions in adjacent order
            temp_mat = temp_mat.reshape(mat.shape[0], mat.shape[1], end_flag, self.array_size[0]).transpose(1, 2)
            if bw_e:
                # define the FP_map_tensor function
                data_int, mat_data, max_mat, e_bias = bfp_map_tensor(temp_mat, slice_method)
            else:
                data_int, mat_data, max_mat, e_bias = quant_map_tensor(temp_mat, slice_method)
            # the transpose is used to make the data_int is the same as the input data
            mat_data = mat_data.transpose(1, 2).reshape(mat.shape[0], mat.shape[1], -1)[:, :, :mat.shape[2]]

        else:
            # the difference between the transpose and the non-transpose is the slice direction
            if mat.shape[1] % self.array_size[0] == 0:
                end_flag = int(mat.shape[1] / self.array_size[0])
            else:
                end_flag = int(mat.shape[1] / self.array_size[0]) + 1

            # todo: optimize the following code using inplace operation
            temp_mat = torch.empty((mat.shape[0], end_flag * self.array_size[0], mat.shape[2]), device=mat.device)
            temp_mat[:, :mat.shape[1], :] = mat
            temp_mat[:, mat.shape[1]:, :] = 0

            temp_mat = temp_mat.reshape(mat.shape[0], end_flag, self.array_size[0], mat.shape[2])
            if bw_e:
                # define the FP_map_tensor function
                data_int, mat_data, max_mat, e_bias = bfp_map_tensor(temp_mat, slice_method, bw_e)
            else:
                data_int, mat_data, max_mat, e_bias = quant_map_tensor(temp_mat, slice_method)
            mat_data = mat_data.reshape(mat.shape[0], -1, mat.shape[2])[:, :mat.shape[1], :]

        # remove the unsqueezed dimension
        if unsqueezed:
            data_int = data_int.squeeze(0)
            mat_data = mat_data.squeeze(0)
            max_mat = max_mat.squeeze(0)
        return data_int, mat_data, max_mat, e_bias

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
    # test for INT format
    if tb_mode == 0:
        torch.manual_seed(42)
        device = torch.device('cuda:0')
        x_data = torch.randn(4000, 1000, device=device)
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
        print("Tensor time: ", end - start)
        # result = engine._test(x, mat)
        result = result.cpu().numpy()

        rel_result = torch.matmul(x_data, mat_data).cpu().numpy()

        print(RE(result, rel_result))
        plt.scatter(rel_result.reshape(-1), result.reshape(-1))
        plt.xlabel('Expected Value of Dot Product')
        plt.ylabel('Measured Value of Dot Product')
        plt.show()
    # test for FP format
    elif tb_mode == 1:
        torch.manual_seed(42)
        device = torch.device('cuda:0')
        x_data = torch.randn(4000, 1000, device=device)
        mat_data = torch.randn(1000, 1200, device=device)
        xblk = torch.tensor([1, 1, 4, 4])
        mblk = torch.tensor([1, 1, 4, 4])
        mat = SlicedData(mblk, device=device, bw_e=8)
        x = SlicedData(xblk, device=device, bw_e=8)

        engine = DPETensor(var=0.0)
        x.slice_data_imp(engine, x_data, transpose=True)
        mat.slice_data_imp(engine, mat_data)
        start = time.time()
        result = engine(x, mat)
        end = time.time()
        print("Tensor time: ", end - start)
        # result = engine._test(x, mat)
        result = result.cpu().numpy()

        rel_result = torch.matmul(x_data, mat_data).cpu().numpy()

        print(RE(result, rel_result))
        plt.scatter(rel_result.reshape(-1), result.reshape(-1))
        plt.xlabel('Expected Value of Dot Product')
        plt.ylabel('Measured Value of Dot Product')
        plt.show()
