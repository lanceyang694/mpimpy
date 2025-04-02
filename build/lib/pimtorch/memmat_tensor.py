# -*- coding:utf-8 -*-
# @File  : dpe_tensor.py
# @Author: Zhou
# @Date  : 2024/6/27

'''
this is a new version of the dpe_tensor.py
we use the tensor to realize the dot product, and only consider the INT format data
this version is more efficient than the previous version
'''
import math

import torch
from matplotlib import pyplot as plt
from utils import SNR, dot_high_dim
from data_formats import SlicedData

import time


def wire_resistance(x, y, wire_resistance):
    pass


class DPETensor(object):
    '''
    Implements a dot product engine using bit-sliced tensor operations for matrix multiplication.
    Supports INT and FP data formats with configurable quantization granularity and device settings.
    '''
    def __init__(
            self, HGS=1e-5, LGS=1e-7, g_level=16, var=0.05, vnoise=0.05, wire_resistance=0,
            rdac=2 ** 4, radc=2 ** 8, vread=0.1,
            stuck_fault_ratio=0, stuck_fault_value=0,
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """
        Parameters:
            HGS (float): High conductance state
            LGS (float): Low conductance state
            g_level (int): Number of conductance levels
            var (float): Random Gaussian noise of conductance
            vnoise (float): Random Gaussian noise of voltage
            wire_resistance (float): Wire resistance
            rdac (int): Number of DAC resolution
            radc (int): Number of ADC resolution
            vread (float): Read voltage
            weight_quant_gran (str or tuple): Quantization granularity of the weight matrix
                "per-matrix" -> The whole matrix is quantized together (i.e., the quantization granularity is (m, n)
                                the same as the matrix shape).
                "per-row" -> Each row of the matrix is quantized separately. (i.e., the quantization granularity is (1, n)).
                "per-col" -> Each column of the matrix is quantized separately. (i.e., the quantization granularity is (m, 1)).
                (a, b) -> The quantization granularity is (a, b).
            input_quant_gran (str or tuple): Quantization granularity of the input matrix
            input_paral_size (tuple): The size of the input matrix used for parallel computation
            weight_paral_size (tuple): The size of the weight matrix used for parallel computation
            stuck_fault_ratio (float): Ratio of stuck faults, range: [0, 1]
            stuck_fault_value (int): Value of stuck faults, 0 or 1 in the current version
        """
        self.HGS = HGS
        self.LGS = LGS
        self.g_level = g_level
        self.var = var
        self.vnoise = vnoise
        self.rdac = rdac
        self.radc = radc
        self.vread = vread

        # these parameters are optional
        self.wire_resistance = wire_resistance
        self.stuck_fault_ratio = stuck_fault_ratio
        self.stuck_fault_value = stuck_fault_value

        self.device = device

        if self.stuck_fault_value == 0:
            self.stuck_fault_value = self.LGS
        elif self.stuck_fault_value == 1:
            self.stuck_fault_value = self.HGS
        else:
            raise ValueError('The stuck fault value should be 0 or 1!, other values are not supported in current version!')
        if self.radc < 2:
            raise ValueError('The resolution of the ADC should be larger than 1!')
        if self.rdac < 2:
            raise ValueError('The resolution of the DAC should be larger than 1!')
        if self.g_level < 2:
            raise ValueError('The number of the conductance levels should be larger than 1!')
        if self.LGS >= self.HGS:
            raise ValueError('The low conductance state should be smaller than the high conductance state!')

    def __call__(self, x: SlicedData, mat: SlicedData):
        return self.MapReduceDot(x, mat)

    def MapReduceDot(self, x: SlicedData, mat: SlicedData):
        """
        Implements matrix multiplication using the MapReduce method.

        Parameters:
            x (SlicedData): Input tensor (shape: (m, n) or (batch, m, n)).
            mat (SlicedData): Weight tensor (shape: (n, p)).
            wire_factor (bool): Consider wire resistance (not implemented).

        Returns:
            torch.Tensor: Result of the matrix multiplication.
        """
        if mat.device.type != x.device.type:
            raise ValueError('The input data and weight data should be in the same device!')
        # check the quantization shape of the input data and weight data
        if x.shape[-1] != mat.shape[-2]:
            raise ValueError('The input data mismatches the shape of weight data!')
        if self.wire_resistance > 0:
            raise NotImplementedError('The wire_factor is not supported in the training version!')
        else:
            result = self._dot(x, mat, self._num2V, self._gen_read_noise)
        return result

    def _num2G(self, data, max_weights):
        """
        Converts weight data to static resistance.
        Parameters:
            data (torch.Tensor): Weight data.
            max_weights (torch.Tensor): Maximum weight values.

        Returns:
            torch.Tensor: conductance values.
        """
        Q_G = (self.HGS - self.LGS) / (self.g_level - 1)
        G = torch.round(data / max_weights * (self.g_level - 1)) * Q_G + self.LGS

        # todo add static stuck fault
        # two methods to add the stuck fault:
        # 1. add the stuck fault to the weight data before add read noise (this version realization)
        # 2. add the stuck fault to the weight data after add read noise (to verify in the future version)
        if self.stuck_fault_ratio > 0:
            # generate a random mask to determine the stuck fault
            stuck = torch.randint(0, 100, G.shape, device=mat.device)
            stuck = stuck < self.stuck_fault_ratio * 100
            # apply the stuck fault to the weight data
            G[stuck] = self.stuck_fault_value

        # add the wire resistance
        if self.wire_resistance > 0:
            pass
        return G

    def _gen_read_noise(self, mat: SlicedData):
        """
        Converts weight data to resistance with added normal noise.

        Parameters:
            mat (SlicedData): Weight data.

        Returns:
            torch.Tensor: Resistance values.
        """
        r = torch.exp(torch.normal(0, self.var, mat.G.shape, device=mat.device))
        return mat.G * r

    def _num2V(self, x: SlicedData):
        """
        Converts input data to voltage (scaled by read voltage).

        Parameters:
            x (SlicedData): Input data.

        Returns:
            torch.Tensor: Voltage values.
        """
        xmax = x.sliced_max_weights
        if len(x.shape) == 2:  # without batch, the shape is (num_divide_row_x, num_divide_col_x, num_slice_x, m, n)
            xmax = xmax.reshape(1, 1, -1, 1, 1)
        elif len(x.shape) == 3:  # with batch, the shape is (batch, num_divide_row_x, num_divide_col_x, num_slice_x, m, n)
            xmax = xmax.reshape(1, 1, 1, -1, 1, 1)
        else:
            raise ValueError('The input data dimension is not supported!')
        V_in = self.vread * torch.round(x.sliced_data / xmax * (self.rdac - 1)) / (self.rdac - 1)
        return V_in

    def _dot(self, x: SlicedData, mat: SlicedData, _num2V_func, _num2R_func):
        """
        Computes the dot product of input and weight tensors.

        Parameters:
            x (SlicedData): Input tensor with shape (m, n) or (batch, m, n).
            mat (SlicedData): Weight tensor with shape (n, p).
            _num2V_func (function): Function to convert input data to voltage.
            _num2R_func (function): Function to convert weight data to resistance

        Returns:
            torch.Tensor: Result of the dot product with shape (m, p) or (batch, m, p).
        """
        Vin = _num2V_func(x)
        G = _num2R_func(mat)

        if max(mat.sliced_max_weights) > self.g_level - 1:
            raise ValueError('The weight data is out of the range!')

        if len(x.shape) == 2:
            adcRef = (self.HGS - self.LGS) * self.vread * Vin.shape[-1]
            QG = (self.HGS - self.LGS) / (self.g_level - 1)
            out = dot_high_dim(Vin, G - self.LGS)
            out = torch.round(out / adcRef * (self.radc - 1)) / (self.radc - 1)
            out = torch.mul(out, x.sliced_max_weights.reshape(1, 1, 1, -1, 1, 1, 1))
            out = (torch.mul(out, mat.sliced_max_weights.reshape(1, 1, 1, 1, -1, 1, 1)) / QG / self.vread / (
                        self.g_level - 1) * adcRef)
            shift_weights = torch.zeros((len(x), len(mat)), device=x.device)

            for i in range(len(x)):
                shift_weights[i] = x.sliced_weights[i] * mat.sliced_weights
            out = torch.mul(out.reshape(out.shape[0], out.shape[1], out.shape[2], -1, out.shape[5], out.shape[6]),
                            shift_weights.reshape(1, 1, 1, -1, 1, 1))
            out = out.sum(dim=3)
            if x.bw_e is None:
                out_block_max = torch.einsum("nmij, mpij->nmpij", x.max_data, mat.max_data)
                out = (out * out_block_max / (2 ** (sum(x.slice_method) - 1) - 1) / (
                            2 ** (sum(mat.slice_method) - 1) - 1))
            else:
                out_block_e_bias = torch.einsum("nmij, mpij->nmpij", 2. ** x.e_bias, 2. ** mat.e_bias)
                out = out * out_block_e_bias * 2. ** (4 - sum(x.slice_method) - sum(mat.slice_method))
            out = out.sum(dim=1)
            out = out.permute(0, 2, 1, 3)
            out = out.reshape(out.shape[0] * out.shape[1], out.shape[2] * out.shape[3])
            out = out[:x.shape[0], :mat.shape[1]]
        elif len(x.shape) == 3:  # 三维可能会出错， todo：调试
            adcRef = (self.HGS - self.LGS) * self.vread * Vin.shape[-1]
            QG = (self.HGS - self.LGS) / (self.g_level - 1)
            out = dot_high_dim(Vin, G - self.LGS)
            out = torch.round(out / adcRef * (self.radc - 1)) / (self.radc - 1)
            out = torch.mul(out, x.sliced_max_weights.reshape(1, 1, 1, 1, -1, 1, 1, 1))
            out = (torch.mul(out, mat.sliced_max_weights.reshape(1, 1, 1, 1, 1, -1, 1, 1)) / QG / self.vread / (
                        self.g_level - 1) * adcRef)
            shift_weights = torch.zeros((len(x), len(mat)), device=x.device)

            for i in range(len(x)):
                shift_weights[i] = x.sliced_weights[i] * mat.sliced_weights
            # add the shift weights to the calculated result
            out = torch.mul(out.reshape(out.shape[0], out.shape[1], out.shape[2], out.shape[3], -1, out.shape[6],
                                         out.shape[7]), shift_weights.reshape(1, 1, 1, 1, -1, 1, 1))
            out = out.sum(dim=4)
            if x.bw_e is None:
                out_block_max = torch.einsum("bnmij, mpij->bnmpij", x.max_data, mat.max_data)
                out = (out * out_block_max / (2 ** (sum(x.slice_method) - 1) - 1) / (
                            2 ** (sum(mat.slice_method) - 1) - 1))
            else:
                out_block_e_bias = torch.einsum("bnmij, mpij->bnmpij", 2. ** x.e_bias, 2. ** mat.e_bias)
                out = out * out_block_e_bias * 2. ** (4 - sum(x.slice_method) - sum(mat.slice_method))
            out = out.sum(dim=2)
            out = out.permute(0, 1, 3, 2, 4)
            out = out.reshape(out.shape[0], out.shape[1] * out.shape[2], out.shape[3] * out.shape[4])
            out = out[:out.shape[0], :x.shape[1], :mat.shape[1]]
        else:
            raise ValueError('The input data dimension is not supported!')
        return out

    def _test(self, x: SlicedData, mat: SlicedData):
        temp = dot_high_dim(x.sliced_data.float(), mat.sliced_data.float())
        shift_weights = torch.zeros((len(x), len(mat)), device=x.device)
        for i in range(len(x)):
            shift_weights[i] = x.sliced_weights[i] * mat.sliced_weights
        out = torch.mul(temp.reshape(temp.shape[0], -1, temp.shape[3], temp.shape[4]),
                        shift_weights.reshape(1, -1, 1, 1))
        out = (out.sum(dim=1) * x.max_data * mat.max_data
               / (2 ** (sum(x.slice_method) - 1) - 1) / (2 ** (sum(mat.slice_method) - 1) - 1))
        return out.sum(dim=0)

def c_profile_test():
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    x_data = torch.randn(4000, 500, device=device)
    mat_data = torch.randn(500, 1200, device=device)
    xblk = torch.tensor([1, 1, 4, 4])
    mblk = torch.tensor([1, 1, 4, 4])
    mat = SlicedData(mblk, device=device, bw_e=8)

    engine = DPETensor(var=0.00)
    for i in range(1):
        print(i)
        x = SlicedData(xblk, device=device, bw_e=8)
        x.slice_data_imp(engine, x_data)
        mat.slice_data_imp(engine, mat_data)
        result = engine(x, mat)

if __name__ == '__main__':
    tb_mode = 1
    device = torch.device('cuda:0')
    if tb_mode == 0:
        torch.manual_seed(42)
        x_data = torch.randn(2, 1000, 1000, dtype=torch.float64, device=device)
        mat_data = torch.randn(1000, 1000, dtype=torch.float64, device=device)
        mblk = torch.tensor([1, 1, 2, 4])
        xblk = torch.tensor([1, 1, 2, 4])
        mat = SlicedData(mblk, device=device, bw_e=8, is_weight=True, quant_gran=(64, 64), paral_size=(64, 64))
        x = SlicedData(xblk, device=device, bw_e=8, quant_gran=(64, 64), paral_size=(64, 64))
        engine = DPETensor(var=0.05, g_level=16, rdac=16, radc=2 ** 8)
        mat.slice_data_imp(engine, mat_data)
        x.slice_data_imp(engine, x_data)
        start = time.time()
        result = engine(x, mat).cpu().numpy()
        rel_result = torch.matmul(x_data, mat_data).cpu().numpy()
        snr_varlue = SNR(result, rel_result)
        print("SNR(dB)", snr_varlue)
        plt.scatter(rel_result.reshape(-1), result.reshape(-1))
        plt.xlabel('Expected Value of Dot Product')
        plt.ylabel('Measured Value of Dot Product')
        plt.show()
    elif tb_mode == 1:
        torch.manual_seed(42)
        x_data = torch.randn(1000, 1000, dtype=torch.float64, device=device)
        mat_data = torch.randn(1000, 1000, dtype=torch.float64, device=device)
        mblk = torch.tensor([1, 1, 2, 2])
        xblk = torch.tensor([1, 1, 2, 2])
        size = 64
        paral_size = size
        mat = SlicedData(mblk, device=device, bw_e=None, is_weight=True,
                         paral_size=(paral_size, paral_size), quant_gran=(1,size))
        x = SlicedData(xblk, device=device, bw_e=None, paral_size=(paral_size, paral_size), quant_gran=(1,size))
        engine = DPETensor(var=0.00, g_level=16, rdac=16, radc=2**12)
        mat.slice_data_imp(engine, mat_data)
        x.slice_data_imp(engine, x_data)
        start = time.time()
        result = engine(x, mat).cpu().numpy()
        rel_result = torch.matmul(x_data, mat_data).cpu().numpy()
        snr_varlue = SNR(result, rel_result)
        print("SNR(dB)", snr_varlue)

        plt.scatter(rel_result.reshape(-1), result.reshape(-1))
        plt.xlabel('Expected Value of Dot Product')
        plt.ylabel('Measured Value of Dot Product')
        plt.show()